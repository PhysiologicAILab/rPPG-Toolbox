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
nf = [8, 16, 24, 32]

model_config = {
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



class decoder_block(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug

        # k_t = 3  # 3  # 5   #7
        # pad_t = 1  # 1  # 2   #3
        self.conv_decoder = nn.Sequential(

            nn.Conv3d(nf[3], nf[0], (3, 3, 3), stride=(1, 2, 2), padding=(1, 0, 0)),
            nn.Tanh(),

            nn.Dropout3d(p=dropout_rate),

            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
        )


    def forward(self, x):

        if self.debug:
            print("Decoder")
            print("     x.shape", x.shape)

        x = self.conv_decoder(x)
        
        if self.debug:
            print("     conv_decoder_x.shape", x.shape)
        
        return x



class iBVPNet(nn.Module):
    def __init__(self, frames, device, in_channels=3, dropout=0.2, debug=False):
        super(iBVPNet, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.BatchNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.BatchNorm3d(3)
            self.thermal_norm = nn.BatchNorm3d(1)
        else:
            print("Unsupported input channels")
        
        if self.debug:
            print("nf:", nf)

        self.voxel_embeddings = encoder_block(self.in_channels, dropout_rate=dropout, debug=debug)
        self.decoder = decoder_block(dropout_rate=dropout, debug=debug)

        
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
        
        feats = self.decoder(voxel_embeddings)
        if self.debug:
            print("feats.shape", feats.shape)

        rPPG = feats.view(-1, length-1)

        if self.debug:
            print("rPPG.shape", rPPG.shape)

        return rPPG, voxel_embeddings
    

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import resample
    # from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/iBVPNet')

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
    net = nn.DataParallel(iBVPNet(frames=frames, device=device, in_channels=in_channels, debug=debug)).to(device)
    # net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    if assess_latency:
        time_vec = []
        for passes in range(num_trials):
            t0 = time.time()
            pred, vox_embed = net(test_data)
            t1 = time.time()
            time_vec.append(t1-t0)

        print("Average time: ", np.median(time_vec))
        plt.plot(time_vec)
        plt.show()
    else:
        pred, vox_embed = net(test_data)
    # print("-"*100)
    # print(net)
    # print("-"*100)

    if visualize:
        test_data = test_data.detach().numpy()
        vox_embed = vox_embed.detach().numpy()

        # print(test_data.shape, vox_embed.shape, factorized_embed.shape)
        b, ch, enc_frames, enc_height, enc_width = vox_embed.shape
        # exit()
        for ch in range(vox_embed.shape[1]):
            fig, ax = plt.subplots(9, 2, layout="tight")

            frame = 0
            ax[0, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[0, 0].axis('off')
            ax[0, 1].imshow(vox_embed[0, ch, frame, :, :])
            ax[0, 1].axis('off')

            frame = 20
            ax[1, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[1, 0].axis('off')
            ax[1, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[1, 1].axis('off')

            frame = 40
            ax[2, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[2, 0].axis('off')
            ax[2, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[2, 1].axis('off')

            frame = 60
            ax[3, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[3, 0].axis('off')
            ax[3, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[3, 1].axis('off')

            frame = 80
            ax[4, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[4, 0].axis('off')
            ax[4, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[4, 1].axis('off')

            frame = 100
            ax[5, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[5, 0].axis('off')
            ax[5, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[5, 1].axis('off')

            frame = 120
            ax[6, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[6, 0].axis('off')
            ax[6, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[6, 1].axis('off')

            frame = 140
            ax[7, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[7, 0].axis('off')
            ax[7, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[7, 1].axis('off')

            frame = 159
            ax[8, 0].imshow(np_data[frame, ...].astype(np.uint8))
            ax[8, 0].axis('off')
            ax[8, 1].imshow(vox_embed[0, ch, frame//4, :, :])
            ax[8, 1].axis('off')

            plt.show()
            plt.close(fig)

    print("pred.shape", pred.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total parameters = ", pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters = ", pytorch_trainable_params)

    # writer.add_graph(net, test_data)
    # writer.close()