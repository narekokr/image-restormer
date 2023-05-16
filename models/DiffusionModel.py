import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_blocks=5, num_diffusion_steps=100):
        super(DiffusionModel, self).__init__()

        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.num_diffusion_steps = num_diffusion_steps

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ))

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        # Normalize input
        x = x / 255.0

        # Encoder
        enc = self.encoder(x)

        # Diffusion
        for i in range(self.num_diffusion_steps):
            noise = torch.randn_like(x)
            x_noisy = mask * x + (1 - mask) * noise
            enc_noisy = self.encoder(x_noisy)
            for j in range(self.num_blocks):
                enc_noisy = self.res_blocks[j](enc_noisy)
            dec = self.decoder(enc)
            x = x + (1.0 / self.num_diffusion_steps) * (dec - x)
        x_diffused = x

        # Decoder
        # out = self.decoder(x_diffused)

        # Unnormalize output
        # out = torch.clamp(out * 255.0, 0, 255)

        # Mask output
        # out = mask * x + (1 - mask) * out

        return x_diffused
