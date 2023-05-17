import torch
from torch import nn
import torch.nn.functional as F

from models.PartialConv2D import PartialConv2d


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class PartialConvUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            PartialConv2d(in_channels, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PartialConv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            PartialConv2d(128, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        res_blocks = []
        for i in range(3):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        masked_x = x * (1 - mask)
        x = self.encoder(masked_x)
        for i in range(3):
            x = self.res_blocks[i](x)
        x = self.decoder(x)

        return x * mask + masked_x