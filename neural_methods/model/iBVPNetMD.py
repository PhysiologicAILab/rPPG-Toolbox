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
nf = [8, 16, 16, 16]

model_config = {
    "MD_R": 3,
    "MD_S": 5,
    "MD_STEPS": 3,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "MD_TYPE": "NMF",
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": 8,
    "height": 72,
    "weight": 72,
    "batch_size": 2,
    "frames": 160,
    "debug": True,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_iBVPNetMD_FactorizePhys_5_Epoch26.pth",
    "data_path": "/Users/jiteshjoshi/Downloads/rPPG_Testing/data/1003_input13.npy",
    "label_path": "/Users/jiteshjoshi/Downloads/rPPG_Testing/data/1003_label3.npy"
}

class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

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

            # channels and spatial elements are considered as features
            # while each time point is considered as a sample
            # D = C * H * W // self.S
            # N = T                   

            # D = C // self.S
            # N = T * H * W

            # # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            # # From spatial and channel dimension, which are are examples, only 2-4 shall be enough to generate the approximated attention matrix
            D = T // self.S
            N = C * H * W 
            # self.R = max(4, min(D//8, N // 32))
            # self.R = max(4, min(D, N) // 8)

            # D = T * H * W // self.S
            # N = C

            # D = T * C // self.S
            # N = H * W

            # D = T * C // self.S
            # N = H * W

            x = x.view(B * self.S, D, N)

            if self.debug:
                print("C, T, H, W", C, T, H, W)
                print("MD_S", self.S)
                print("MD_D", D)
                print("MD_N", N)
                print("MD_R", self.R)
                print("MD_TRAIN_STEPS", self.train_steps)
                print("MD_EVAL_STEPS", self.eval_steps)
                print("x.view(B * self.S, D, N)", x.shape)

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
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
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
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
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
    def __init__(self, inC, device, md_config, debug=False):
        super().__init__()

        self.device = device
        md_type = model_config["MD_TYPE"]
        align_C = model_config["align_channels"] #in_c // 2  # // 2 #// 8

        if "nmf" in md_type.lower():
            self.pre_conv_block = nn.Sequential(
                nn.Conv3d(inC, align_C, (1, 1, 1)),
                nn.ReLU(inplace=True)
            )
        else:
            self.pre_conv_block = nn.Conv3d(align_C, align_C, (1, 1, 1))

        if "nmf" in md_type.lower():
            self.md_block = NMF(self.device, md_config, debug=debug)
        elif "vq" in md_type.lower():
            self.md_block = VQ(self.device, md_config, debug=debug)
        else:
            print("Unknown type specified for MD_TYPE:", md_type)
            exit()

        self.post_conv_block = nn.Sequential(
            ConvBNReLU(align_C, align_C, kernel_size=(1, 1, 1)),
            nn.Conv3d(align_C, inC, (1, 1, 1), bias=False)
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
        att = self.md_block(x)
        dist = torch.dist(x, att)
        att = self.post_conv_block(att)

        x = torch.multiply(shortcut, att)

        return x, att, dist

    def online_update(self, bases):
        if hasattr(self.md_block, 'online_update'):
            self.md_block.online_update(bases)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class encoder_block(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(encoder_block, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug

        self.encoder = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 2, 2], [1, 1, 1]),
            ConvBlock3D(nf[3], nf[3], [3, 3, 3], [1, 1, 1], [1, 0, 0]),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.debug:
            print("Encoder")
            print("     x.shape", x.shape)
        return x



class BVP_Head(nn.Module):
    def __init__(self, md_config, device, use_fsam=True, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = use_fsam

        if self.use_fsam:
            inC = nf[3]
            # self.align_feats = nn.Sequential(
            #     nn.Conv3d(inC, model_config["align_channels"], (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            #     nn.Tanh(),            
            # )
            self.VEFM = FeaturesFactorizationModule(inC, device, md_config, debug=debug)

            # inC = 2 * model_config["align_channels"]
        else:
            inC = nf[3]

        self.conv_decoder = nn.Sequential(

            nn.Conv3d(inC, nf[0], (3, 3, 3), stride=(1, 2, 2), padding=(1, 0, 0)),
            nn.Tanh(),

            nn.Dropout3d(p=dropout_rate),

            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
        )

    def forward(self, voxel_embeddings):

        if self.debug:
            print("Decoder")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if self.use_fsam:
            # voxel_embeddings = self.align_feats(voxel_embeddings)
            factorized_embeddings, att_mask, appx_error = self.VEFM(voxel_embeddings)
            if self.debug:
                print("factorized_embeddings.shape", factorized_embeddings.shape)

            # If residual connection is used, factorization should aim at very low rank approximation to retain only highly important features.
            x = voxel_embeddings + factorized_embeddings

            # # In this case (no residual connection), factorization should aim at optimal rank approximation,
            # # eliminating only some features, while retaining the most
            # x = factorized_embeddings

            # # Concatenate
            # x = torch.cat([voxel_embeddings, factorized_embeddings], dim=1)

        x = self.conv_decoder(x)
        
        if self.debug:
            print("     conv_decoder_x.shape", x.shape)
        
        if self.use_fsam:
            return x, factorized_embeddings, att_mask, appx_error
        else:
            return x



class iBVPNetMD(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.2, use_fsam=True, device=torch.device("cpu"), debug=False):
        super(iBVPNetMD, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.BatchNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.BatchNorm3d(3)
            self.thermal_norm = nn.BatchNorm3d(1)
        else:
            print("Unsupported input channels")
        
        self.use_fsam = use_fsam

        if self.debug:
            print("nf:", nf)

        self.voxel_embeddings = encoder_block(self.in_channels, dropout_rate=dropout, debug=debug)

        self.rppg_head = BVP_Head(md_config, device=device, use_fsam=use_fsam, dropout_rate=dropout, debug=debug)


        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape
        
        # if self.in_channels == 1:
        #     x = x[:, :, :-1, :, :]
        # else:
        #     x = torch.diff(x, dim=2)
        
        x = torch.diff(x, dim=2)

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

        voxel_embeddings = self.voxel_embeddings(x)
        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
        
        if self.use_fsam:
            rppg_feats, factorized_embeddings, att_mask, appx_error = self.rppg_head(voxel_embeddings)
        else:
            rppg_feats = self.rppg_head(voxel_embeddings)

        if self.debug:
            print("rppg_feats.shape", rppg_feats.shape)

        rPPG = rppg_feats.view(-1, length-1)

        if self.debug:
            print("rPPG.shape", rPPG.shape)

        if self.use_fsam:
            return rPPG, voxel_embeddings, factorized_embeddings, att_mask, appx_error
        else:
            return rPPG, voxel_embeddings
    

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import resample
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/iBVPNetMD')

    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_iBVPNetMD_FactorizePhys_5_Epoch6.pth"
    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_iBVPNetMD_FactorizePhys_5_Epoch12.pth"
    # ckpt_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/models/PURE_PURE_iBVP_iBVPNetMD_FactorizePhys_5_Epoch19.pth"
    ckpt_path = model_config["ckpt_path"]

    # data_path = "/Users/jiteshjoshi/Downloads/rPPG_Testing/data/subject36_input9.npy"
    data_path = model_config["data_path"]

    label_path = model_config["label_path"]


    batch_size = model_config["batch_size"]
    frames = model_config["frames"]
    in_channels = model_config["in_channels"]
    data_channels = model_config["data_channels"]
    height = model_config["height"]
    width = model_config["weight"]
    debug = bool(model_config["debug"])
    assess_latency = bool(model_config["assess_latency"])
    num_trials = model_config["num_trials"]
    visualize = model_config["visualize"]

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    if visualize:
        np_data = np.load(data_path)
        np_label = np.load(label_path)
        np_label = np.expand_dims(np_label, 0)

        print("Chunk data shape", np_data.shape)
        print("Chunk label shape", np_label.shape)
        print("Min Max of input data:", np.min(np_data), np.max(np_data))
        # exit()

        test_data = np.transpose(np_data, (3, 0, 1, 2))
        test_data = torch.from_numpy(test_data)
        test_data = test_data.unsqueeze(0)

        last_frame = torch.unsqueeze(test_data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
        test_data = torch.cat((test_data, last_frame), 2)
    else:
        # test_data = torch.rand(batch_size, in_channels, frames, height, width).to(device)
        test_data = torch.rand(batch_size, data_channels, frames + 1, height, width).to(device)

    test_data = test_data.to(torch.float32).to(device)
    # print(test_data.shape)
    # exit()
    md_config = {}
    md_config["MD_S"] = model_config["MD_S"]
    md_config["MD_R"] = model_config["MD_R"]
    md_config["MD_STEPS"] = model_config["MD_STEPS"]
    net = nn.DataParallel(iBVPNetMD(frames=frames, md_config=md_config, device=device, in_channels=in_channels, debug=debug)).to(device)
    # net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    if assess_latency:
        time_vec = []
        appx_error_list = []
        for passes in range(num_trials):
            t0 = time.time()
            pred, vox_embed, factorized_embed, att_mask, appx_error = net(test_data)
            t1 = time.time()
            time_vec.append(t1-t0)
            appx_error_list.append(appx_error.item())

        print("Average time: ", np.mean(time_vec))
        print("Average error:", np.mean(appx_error_list))
        plt.plot(time_vec)
        plt.show()
    else:
        pred, vox_embed, factorized_embed, att_mask, appx_error = net(test_data)
    
    print("Appx error: ", appx_error.item())    #.detach().numpy())
    # print("-"*100)
    # print(net)
    # print("-"*100)

    if visualize:
        test_data = test_data.detach().numpy()
        vox_embed = vox_embed.detach().numpy()
        factorized_embed = factorized_embed.detach().numpy()
        att_mask = att_mask.detach().numpy()

        # print(test_data.shape, vox_embed.shape, factorized_embed.shape)
        b, ch, enc_frames, enc_height, enc_width = vox_embed.shape
        # exit()
        for ch in range(vox_embed.shape[1]):
            fig, ax = plt.subplots(9, 4, layout="tight")

            frame = 0
            ax[0, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[0, 0].axis('off')
            ax[0, 1].imshow(vox_embed[0, ch, frame, :, :])
            ax[0, 1].axis('off')
            ax[0, 2].imshow(factorized_embed[0, ch, frame, :, :])
            ax[0, 2].axis('off')
            ax[0, 3].imshow(att_mask[0, ch, frame, :, :])
            ax[0, 3].axis('off')

            frame = 20
            ax[1, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[1, 0].axis('off')
            ax[1, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[1, 1].axis('off')
            ax[1, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[1, 2].axis('off')
            ax[1, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[1, 3].axis('off')

            frame = 40
            ax[2, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[2, 0].axis('off')
            ax[2, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[2, 1].axis('off')
            ax[2, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[2, 2].axis('off')
            ax[2, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[2, 3].axis('off')

            frame = 60
            ax[3, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[3, 0].axis('off')
            ax[3, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[3, 1].axis('off')
            ax[3, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[3, 2].axis('off')
            ax[3, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[3, 3].axis('off')

            frame = 80
            ax[4, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[4, 0].axis('off')
            ax[4, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[4, 1].axis('off')
            ax[4, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[4, 2].axis('off')
            ax[4, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[4, 3].axis('off')

            frame = 100
            ax[5, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[5, 0].axis('off')
            ax[5, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[5, 1].axis('off')
            ax[5, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[5, 2].axis('off')
            ax[5, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[5, 3].axis('off')

            frame = 120
            ax[6, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[6, 0].axis('off')
            ax[6, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[6, 1].axis('off')
            ax[6, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[6, 2].axis('off')
            ax[6, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[6, 3].axis('off')

            frame = 140
            ax[7, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[7, 0].axis('off')
            ax[7, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[7, 1].axis('off')
            ax[7, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[7, 2].axis('off')
            ax[7, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[7, 3].axis('off')

            frame = 159
            ax[8, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[8, 0].axis('off')
            ax[8, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[8, 1].axis('off')
            ax[8, 2].imshow(factorized_embed[0, ch, frame//4, :, :])
            ax[8, 2].axis('off')
            ax[8, 3].imshow(att_mask[0, ch, frame//4, :, :])
            ax[8, 3].axis('off')

            plt.show()
            plt.close(fig)

    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()