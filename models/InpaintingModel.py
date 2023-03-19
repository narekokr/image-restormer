import torch
import torch.nn as nn


class InpaintingModel(nn.Module):
    def __init__(self, num_blocks=5, num_channels=64):
        super(InpaintingModel, self).__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(num_channels * 4, num_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels * 4, num_channels * 4, kernel_size=3, stride=1, padding=1)
            ))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        # mask = mask.unsqueeze(1)
        x_masked = x * mask
        # x_concat = torch.cat((x_masked, mask), dim=1)
        print(mask.size(), x_masked.size())
        enc = self.encoder(x_masked)
        for i in range(self.num_blocks):
            enc = enc + self.res_blocks[i](enc)
        dec = self.decoder(enc)
        print(dec.size(), x_masked.size(), enc.size())
        output = dec * (1 - mask) + x_masked
        return output * 255.0



class Autoencoder(nn.Module):
    def __init__(self, num_res_blocks=5):
        super(Autoencoder, self).__init__()
        self.num_res_blocks = num_res_blocks

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # Encode
        enc = self.encoder(x)

        # Residual blocks
        for i in range(self.num_res_blocks):
            enc = self.res_blocks[i](enc)

        # Decode
        dec = self.decoder(enc)

        # Masked image
        x_masked = x * mask

        # Inpainting
        x_inpaint = dec * (1 - mask) + x_masked

        return x_inpaint



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
