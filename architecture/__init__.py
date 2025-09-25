import torch
from .MPRNet import MPRNet
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """3D残差块，包含两个卷积层和一个跳跃连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = out + self.shortcut(residual)  # safer

        out = self.leaky_relu(out)

        return out


class ResidualUNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualUNet, self).__init__()

        self.encoder1 = ResidualBlock(in_channels, 32)
        self.encoder2 = ResidualBlock(32, 64)


        self.decoder1 = ResidualBlock(64, 32)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)


        x3 = self.decoder1(x2)
        out = self.decoder2(x3)

        return out


class NetworkWithUNet(nn.Module):
    def __init__(self, original_network, in_channels, out_channels):
        super(NetworkWithUNet, self).__init__()
        self.unet = ResidualUNet(in_channels, out_channels)
        self.original_network = original_network

    def forward(self, x):

        x = self.unet(x)  # 先通过 UNet 1.28.121.16.16
        return self.original_network(x)  # 再通过原始网络



def model_generator(method, pretrained_model_path=None):
    if method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'edsr':
        model = EDSR().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
    elif method == 'hrnet':
        model = SGN().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'awan':
        model = AWAN().cuda()
    elif method == 'unet':
        model = UNet().cuda()
        return model

    else:
        print(f'Method {method} is not defined !!!!')
    model = NetworkWithUNet(model, in_channels=3, out_channels=28)

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
