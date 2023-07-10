import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            InPlaceABN(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            InPlaceABN(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            InPlaceABN(out_channels),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = F.interpolate
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.up(x1, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.down4 = (Down(64, 128 ))
        self.up1 = (Up(128, 64 , bilinear))
        self.up2 = (Up(64, 32 , bilinear))
        self.up3 = (Up(32, 16 , bilinear))
        self.up4 = (Up(16, 8, bilinear))
        self.outc1 = (OutConv(8, n_classes))
        self.outc2 = (OutConv(n_classes, n_classes))


        
        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.outc1(x)
        logits = self.outc2(feature)

        # original FeatureNet used in depth estimation
        feat2 = self.toplayer(x3)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(x2))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(x1))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        output = {"level_0": feat0, "level_1": feat1, "level_2": feat2, 'logits': logits, 'feature': feature}
                  
        return output


# if __name__ == '__main__':
#     x = torch.randn(6, 3, 256, 256)
#     model = UNet(3, 1)
#     y = model(x)
#     print(y.shape)
