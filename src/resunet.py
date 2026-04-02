import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ResidualBlock(nn.Module):
    """
    残差块：包含两个3x3卷积和一个残差连接
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResidualBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut connection
        # 如果输入输出通道不一致，使用1x1卷积对齐维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Down(nn.Module):
    """
    下采样：池化 + 残差块
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.res_block(x)
        return x


class Up(nn.Module):
    """
    上采样：插值/转置卷积 + 特征拼接 + 残差块
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 拼接后通道数为 in_channels，输出为 out_channels
            # 这里按照原代码习惯设置中间通道
            self.conv = ResidualBlock(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 补齐因奇数尺寸导致的 shape 差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 第一层卷积通常不带残差连接，或使用简单的 ResidualBlock
        self.in_conv = ResidualBlock(in_channels, base_c)

        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return logits


if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=3, num_classes=2, base_c=32).to(device)

    # 模拟输入 [Batch, Channel, Height, Width]
    x = torch.randn(1, 3, 256, 256).to(device)
    output = model(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")  # 预期: [1, 2, 256, 256]