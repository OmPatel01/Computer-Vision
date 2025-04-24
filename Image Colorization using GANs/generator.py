import torch
import torch.nn as nn

# Deep Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Complex Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # ------------- Encoder (8 levels) -------------
        self.enc1 = self._encoder_block(1, 64, bn=False)     # 64x
        self.enc2 = self._encoder_block(64, 128)             # 128x
        self.enc3 = self._encoder_block(128, 256)            # 256x
        self.enc4 = self._encoder_block(256, 512)            # 512x
        self.enc5 = self._encoder_block(512, 1024)           # 1024x
        self.enc6 = self._encoder_block(1024, 1536)          # 1536x
        self.enc7 = self._encoder_block(1536, 1792)          # 1792x
        self.enc8 = self._encoder_block(1792, 2048)          # 2048x

        # ----------- Deep Bottleneck ------------
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(2048) for _ in range(7)]
        )

        # ------------- Decoder (8 levels) -------------
        self.dec8 = self._decoder_block(2048, 1792)
        self.dec7 = self._decoder_block(3584, 1536)
        self.dec6 = self._decoder_block(3072, 1024)
        self.dec5 = self._decoder_block(2048, 512)
        self.dec4 = self._decoder_block(1024, 256)
        self.dec3 = self._decoder_block(512, 128)
        self.dec2 = self._decoder_block(256, 64)

        # Final upsample to output 3-channel RGB image
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Use Sigmoid() if input is scaled [0,1], Tanh() for [-1,1]
        )

    def _encoder_block(self, in_channels, out_channels, bn=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Bottleneck
        b = self.res_blocks(e8)

        # Decoder path with skip connections
        d8 = self.dec8(b)
        d7 = self.dec7(torch.cat([d8, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return d1
