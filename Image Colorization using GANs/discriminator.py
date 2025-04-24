import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def conv_block(in_c, out_c, stride=2, use_bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(4, 64, use_bn=False),   # Input: [B, 4, 128, 128] → [B, 64, 64, 64]
            conv_block(64, 128),               # → [B, 128, 32, 32]
            conv_block(128, 256),              # → [B, 256, 16, 16]
            conv_block(256, 512),              # → [B, 512, 8, 8]
            conv_block(512, 512),              # → [B, 512, 4, 4]
            conv_block(512, 512, stride=1),    # → [B, 512, 4, 4] (same size, more depth)

            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)  # → [B, 1, 4, 4]
        )

    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)  # Shape: [B, 4, H, W]
        return self.model(x)
