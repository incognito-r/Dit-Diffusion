import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """Lightweight discriminator (similar to Pix2Pix)"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # Downsample by 2
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Downsample by 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            # Downsample by 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # Final 1x1 patch output
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)