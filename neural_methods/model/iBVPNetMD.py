import torch.nn as nn
import torch.nn.functional as F



class _MatrixDecomposition3DBase(nn.Module):
    def __init__(self, device, model_config, dim="3D"):
        super().__init__()

        self.dim = dim
        self.S = model_config["model_params"]["MD_S"]
        self.D = model_config["model_params"]["MD_D"]
        self.R = model_config["model_params"]["MD_R"]

        self.train_steps = model_config["model_params"]["TRAIN_STEPS"]
        self.eval_steps = model_config["model_params"]["EVAL_STEPS"]

        self.inv_t = model_config["model_params"]["INV_T"]
        self.eta = model_config["model_params"]["ETA"]

        self.rand_init = model_config["model_params"]["RAND_INIT"]
        self.device = device

        # print('spatial', self.dim)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        
        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        
        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        else:                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = C // self.S
            N = L
            x = x.view(B * self.S, D, N)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.dim:
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, H, W)
        else:
            # (B * S, D, N) -> (B, C, L)
            x = x.view(B, C, L)

        # (B * L, D, R) -> (B, L, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)



class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DeConvBlock3D, self).__init__()

        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, mid_channel, (4, 1, 1), (2, 1, 1), (1, 0, 0)),
            nn.Conv3d(mid_channel, out_channel, (7, 3, 3), (1, 2, 2), (3, 1, 1)),
            nn.BatchNorm3d(out_channel),
            nn.ELU()
        )

    def forward(self, x):
        return self.deconv_block_3d(x)

# num_filters
nf = [8, 16, 24, 40, 64]

class encoder_block(nn.Module):
    def __init__(self, in_channel, debug=False):
        super(encoder_block, self).__init__()
        # in_channel, out_channel, kernel_size, stride, padding

        self.debug = debug
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 1, 1]),

            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 2, 2], [1, 1, 1]),

            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [2, 2, 2], [5, 1, 1]),

            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [2, 1, 1], [5, 1, 1]),

            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Encoder")
            print("x.shape", x.shape)
        st_x = self.spatio_temporal_encoder(x)
        if self.debug:
            print("st_x.shape", st_x.shape)
        t_x = self.temporal_encoder(st_x)
        if self.debug:
            print("t_x.shape", t_x.shape)
        return t_x


class decoder_block(nn.Module):
    def __init__(self, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug
        self.decoder_1 = DeConvBlock3D(nf[4], nf[3], nf[2])
        self.decoder_2 = DeConvBlock3D(nf[2], nf[1], nf[0])

    def forward(self, x):
        if self.debug:
            print("Decoder")
        x = self.decoder_1(x)
        if self.debug:
            print("decoder_1_x.shape", x.shape)
        x = self.decoder_2(x)
        if self.debug:
            print("decoder_2_x.shape", x.shape)
        return x



class iBVPNetMD(nn.Module):
    def __init__(self, frames, in_channels=3, debug=False):
        super(iBVPNetMD, self).__init__()
        self.debug = debug
        self.model = nn.Sequential(
            encoder_block(in_channels, debug),
            decoder_block(debug),
            # spatial adaptive pooling
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(nf[0], 1, [1, 1, 1], stride=1, padding=0)
        )

        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape
        feats = self.model(x)
        if self.debug:
            print("feats.shape", feats.shape)
        rPPG = feats.view(-1, length)
        return rPPG
    

if __name__ == "__main__":
    import torch
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/iBVPNetMD')

    # duration = 8
    # fs = 25
    batch_size = 1
    frames = 128    #duration*fs
    in_channels = 1
    height = 64
    width = 64
    test_data = torch.rand(batch_size, in_channels, frames, height, width)

    net = iBVPNetMD(in_channels=in_channels, frames=frames, debug=True)
    # print("-"*100)
    # print(net)
    # print("-"*100)

    pred = net(test_data)
    print("pred.shape", pred.shape)

    # writer.add_graph(net, test_data)
    # writer.close()
