import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ResidualBlock(nn.Module):
    """
    复用残差块：包含两个3x3卷积和一个残差连接
    """

    def __init__(self, in_channels, out_channels, stride=1, mid_channels=None):
        super(ResidualBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ASPPModule(nn.Module):
    """
    ASPP 模块：空洞空间卷积池化金字塔
    """

    def __init__(self, in_channels: int, out_channels: int, rates: List[int]):
        super(ASPPModule, self).__init__()

        # 1. 1x1 卷积分支
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 不同扩张率的 3x3 卷积分支
        self.modules_list = nn.ModuleList()
        for rate in rates:
            self.modules_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 3. 全局平均池化分支
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 4. 融合层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [self.b0(x)]
        for stage in self.modules_list:
            res.append(stage(x))

        # 全局池化分支需要上采样还原尺寸
        gp = self.global_pooling(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gp)

        out = torch.cat(res, dim=1)
        return self.bottleneck(out)


class DeepLabV3Plus(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 32):
        super(DeepLabV3Plus, self).__init__()

        # Encoder (Backbone: ResNet-like)
        # 产生低级特征 (Low-level features)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            ResidualBlock(base_c, base_c)
        )  # 1/1

        self.stage2 = ResidualBlock(base_c, base_c * 2, stride=2)  # 1/2
        self.stage3 = ResidualBlock(base_c * 2, base_c * 4, stride=2)  # 1/4 (用于跳连的浅层特征)

        # 产生高级语义特征
        self.stage4 = ResidualBlock(base_c * 4, base_c * 8, stride=2)  # 1/8
        self.stage5 = ResidualBlock(base_c * 8, base_c * 16, stride=2)  # 1/16

        # ASPP 提取多尺度上下文
        self.aspp = ASPPModule(base_c * 16, 256, rates=[6, 12, 18])

        # Decoder
        # 对浅层特征进行 1x1 卷积降维
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(base_c * 4, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 特征融合与最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        # 编码器提取特征
        x = self.stage1(x)
        x = self.stage2(x)
        low_level_feat = self.stage3(x)  # 1/4 特征用于跳连

        x = self.stage4(low_level_feat)
        x = self.stage5(x)

        # 高级语义特征经过 ASPP
        x = self.aspp(x)  # 1/16

        # 解码器融合
        # 1. 高级特征 4 倍上采样
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)

        # 2. 浅层特征降维
        low_level_feat = self.low_level_conv(low_level_feat)

        # 3. 拼接并卷积
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.final_conv(x)

        # 最后 4 倍上采样到原图大小
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(in_channels=3, num_classes=2, base_c=32).to(device)

    # 模拟输入 [Batch, Channel, Height, Width]
    x = torch.randn(1, 3, 512, 512).to(device)
    output = model(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")  # 预期: [1, 2, 512, 512]