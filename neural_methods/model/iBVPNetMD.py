"""iBVPNetMD - 3D Convolutional Network with Matrix Decomposition based Feature Distillation.

iBVPNet Model - proposed along with the iBVP Dataset, see https://doi.org/10.3390/electronics13071334

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np

# num_filters
nf = [8, 16, 24, 32, 48, 64]

model_config = {
    "INPUT_CHANNELS": 1,
    "MD_S": 1,
    "TRAIN_STEPS": 6,
    "EVAL_STEPS": 6,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "MD_TYPE": "NMF"
}

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, MD_R, dim="3D"):
        super().__init__()

        self.dim = dim
        self.S = model_config["MD_S"]
        self.R = MD_R

        self.train_steps = model_config["TRAIN_STEPS"]
        self.eval_steps = model_config["EVAL_STEPS"]

        self.inv_t = model_config["INV_T"]
        self.eta = model_config["ETA"]

        self.rand_init = model_config["RAND_INIT"]
        self.device = device

        # print('Dimension:', self.dim)
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
            B, C, T, H, W = x.shape

            # D = C * H * W // self.S # channels and spatial elements are considered as features
            # N = T                   # while each time point is considered as a sample

            # D = C // self.S
            # N = T * H * W

            D = T           # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            N = C * H * W // self.S     # From spatial and channel dimension, which are are examples, only 2-4 shall be enough to generate the approximated attention matrix

            # D = T * H * W // self.S
            # N = C

            # D = C * H // self.S
            # N = T * W

            # D = T * C // self.S
            # N = H * W

            x = x.view(B * self.S, D, N)

            # print("C, T, H, W", C, T, H, W)
            # print("D", D)
            # print("R", self.R)
            # print("N", N)
            # print("x.shape", x.shape)

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

        if self.dim == "3D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
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


class NMF(_MatrixDecompositionBase):
    def __init__(self, device, MD_R, dim="3D"):
        super().__init__(device, MD_R, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class VQ(_MatrixDecompositionBase):
    def __init__(self, device, dim="3D"):
        super().__init__(device, dim=dim)
        self.device = device

    def _build_bases(self, B, S, D, R):
        bases = torch.randn((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        # (B * S, D, N), normalize x along D (for cosine similarity)
        std_x = F.normalize(x, dim=1)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        # normalize along N
        coef = coef / (1e-6 + coef.sum(dim=1, keepdim=True))

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        bases = torch.bmm(x, coef)

        return bases, coef


    def compute_coef(self, x, bases, _):
        with torch.no_grad():
            # (B * S, D, N) -> (B * S, 1, N)
            x_norm = x.norm(dim=1, keepdim=True)

        # (B * S, D, N) / (B * S, 1, N) -> (B * S, D, N)
        std_x = x / (1e-6 + x_norm)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, N, D)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        return coef


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size):
        if kernel_size == (1, 1, 1):
            return (0, 0, 0)
        elif kernel_size == (3, 3, 3):
            return (1, 1, 1)

    def __init__(self, in_c, out_c,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1), padding='same',
                 dilation=(1, 1, 1), groups=1, act='relu', apply_bn=True):
        super().__init__()

        self.apply_bn = apply_bn
        if padding == 'same':
            padding = self._same_paddings(kernel_size)

        self.conv = nn.Conv3d(in_c, out_c,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups,
                              bias=False)
        if self.apply_bn:
            self.bn = nn.BatchNorm3d(out_c)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x)
        x = self.act(x)

        return x


class FeaturesFactorizationModule(nn.Module):
    def __init__(self, device, in_c, frames):
        super().__init__()

        self.device = device
        md_type = model_config["MD_TYPE"]
        mid_C = in_c // 4
        # MD_R = (frames // 4) // 8  # // 4 done by encoder, and //4 for NMF
        MD_R = 8

        if "nmf" in md_type.lower():
            self.pre_conv_block = nn.Sequential(
                nn.Conv3d(in_c, mid_C, (1, 1, 1)),
                nn.ReLU(inplace=True)
            )
        else:
            self.pre_conv_block = nn.Conv3d(in_c, mid_C, (1, 1, 1))

        if "nmf" in md_type.lower():
            self.md_block = NMF(self.device, MD_R)
        elif "vq" in md_type.lower():
            self.md_block = VQ(self.device)
        else:
            print("Unknown type specified for MD_TYPE:", md_type)
            exit()

        self.post_conv_block = nn.Sequential(
            ConvBNReLU(mid_C, mid_C, kernel_size=(1, 1, 1)),
            nn.Conv3d(mid_C, in_c, (1, 1, 1), bias=False)
        )
        self.shortcut = nn.Sequential()
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_conv_block(x)
        x = self.md_block(x)
        x = self.post_conv_block(x)

        x = F.relu(x + shortcut, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.md_block, 'online_update'):
            self.md_block.online_update(bases)



class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channel),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block_3d(x)



class encoder_block(nn.Module):
    def __init__(self, inCh, debug=False):
        super(encoder_block, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        k_t = 7  # 3  # 5   #7
        pad_t = 3  # 1  # 2   #3
        self.debug = debug
        self.encoder = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[0], nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),

            ConvBlock3D(nf[0], nf[0], [k_t, 3, 3], [1, 1, 1], [pad_t, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [k_t, 3, 3], [1, 2, 2], [pad_t, 1, 1]),

            ConvBlock3D(nf[1], nf[1], [k_t, 3, 3], [1, 1, 1], [pad_t, 1, 1]),
            ConvBlock3D(nf[1], nf[2], [k_t, 3, 3], [2, 2, 2], [pad_t, 1, 1]),

            ConvBlock3D(nf[2], nf[2], [k_t, 3, 3], [1, 1, 1], [pad_t, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [k_t, 3, 3], [1, 2, 2], [pad_t, 0, 0]),

            ConvBlock3D(nf[3], nf[3], [k_t, 3, 3], [1, 1, 1], [pad_t, 1, 1]),
            ConvBlock3D(nf[3], nf[4], [k_t, 3, 3], [2, 1, 1], [pad_t, 1, 1]),

            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[4], nf[5], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.debug:
            print("Encoder")
            print("     x.shape", x.shape)
        return x


class DeConvBlock3D(nn.Module):
    def __init__(self, inCh, m1Ch, m2Ch, m3Ch, m4Ch, m5Ch, outCh):
        super(DeConvBlock3D, self).__init__()
        k_t = 7  # 3  # 5   #7
        pad_t = 3  # 1  # 2   #3
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose3d(inCh, m1Ch, (4, 1, 1), (2, 1, 1), (1, 0, 0)),
            nn.BatchNorm3d(m1Ch),
            nn.ELU(),
            nn.Conv3d(m1Ch, m2Ch, (k_t, 3, 3), (1, 2, 2), (pad_t, 1, 1)),
            nn.BatchNorm3d(m2Ch),
            nn.ELU(),
            nn.ConvTranspose3d(m2Ch, m3Ch, (4, 1, 1), (2, 1, 1), (1, 0, 0)),
            nn.BatchNorm3d(m3Ch),
            nn.ELU(),
            nn.Conv3d(m3Ch, m4Ch, (k_t, 3, 3), (1, 2, 2), (pad_t, 1, 1)),
            nn.BatchNorm3d(m4Ch),
            nn.ELU(),
            nn.Conv3d(m4Ch, m5Ch, (k_t, 2, 2), (1, 1, 1), (pad_t, 0, 0)),
            nn.BatchNorm3d(m5Ch),
            nn.ELU(),
            nn.Conv3d(m5Ch, outCh, (k_t, 1, 1), (1, 1, 1), (pad_t, 0, 0)),
        )

    def forward(self, x):
        return self.deconv_block(x)


class decoder_block(nn.Module):
    def __init__(self, device, frames, use_nmf, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug
        self.use_nmf = use_nmf
        # MD_D = nf[1]
        # self.squeeze = ConvBNReLU(nf[5], nf[2], (3, 3, 3), (1, 1, 1))
        if self.use_nmf:
            self.feature_factorizer = FeaturesFactorizationModule(device, nf[5], frames)
        # self.align = ConvBNReLU(nf[2], nf[2], (1, 1, 1))
        self.conv_decoder = DeConvBlock3D(
            nf[5], nf[4], nf[3], nf[2], nf[1], nf[0], 1)  # note: nf[5] = 2 * nf[3]

    def forward(self, x):        

        # short_x = self.squeeze(x)
        if self.use_nmf:
            factorized_x = self.feature_factorizer(x)
        # x = self.align(x)

        if self.debug:
            print("Decoder")
            print("     x.shape", x.shape)
            if self.use_nmf:
                print("     factorized_x.shape", factorized_x.shape)

        # x = self.conv_decoder(torch.concat([short_x, x], dim=1))
        if self.use_nmf:
            x = self.conv_decoder(factorized_x)
        else:
            x = self.conv_decoder(x)
        
        if self.debug:
            print("     conv_decoder_x.shape", x.shape)
        
        return x



class iBVPNetMD(nn.Module):
    def __init__(self, frames, device, in_channels=3, debug=False):
        super(iBVPNetMD, self).__init__()
        self.debug = debug
        use_nmf = True
        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.BatchNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.BatchNorm3d(3)
            self.thermal_norm = nn.BatchNorm3d(1)
        else:
            print("Unsupported input channels")
        self.iBVPNetMD_model = nn.Sequential(
            encoder_block(self.in_channels, debug),
            decoder_block(device, frames, use_nmf, debug)
            # spatial adaptive pooling
            # nn.AdaptiveAvgPool3d((frames, 1, 1)),
            # nn.Conv3d(nf[0], 1, (1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0))
            # nn.Conv3d(nf[0], 1, (1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0))
        )
        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("BatchNormed shape", x.shape)

        feats = self.iBVPNetMD_model(x)
        if self.debug:
            print("feats.shape", feats.shape)
        rPPG = feats.view(-1, length)

        if self.debug:
            print("rPPG.shape", rPPG.shape)

        return rPPG
    

if __name__ == "__main__":
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/iBVPNetMD')

    # duration = 8
    # fs = 25
    batch_size = 2
    frames = 180    #duration*fs
    in_channels = 4
    data_channels = 4
    height = 72
    width = 72

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    # test_data = torch.rand(batch_size, in_channels, frames, height, width).to(device)
    test_data = torch.rand(batch_size, data_channels, frames, height, width).to(device)
    net = iBVPNetMD(frames=frames, device=device, in_channels=in_channels, debug=True)

    # print("-"*100)
    # print(net)
    # print("-"*100)

    pred = net(test_data)
    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()