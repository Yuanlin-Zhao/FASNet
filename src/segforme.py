import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# --- 核心组件 1: Mix-FFN (代替位置编码) ---
class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)

        # 深度卷积：这是 SegFormer 不需要位置编码的关键
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        # Reshape to 2D for DWConv
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


# --- 核心组件 2: Efficient Self-Attention ---
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 序列缩减：降低计算复杂度的关键
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# --- 核心组件 3: Transformer Block ---
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


# --- 核心组件 4: Overlap Patch Merging (下采样) ---
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# --- SegFormer 解码器头 ---
class SegFormerHead(nn.Module):
    def __init__(self, in_channels: List[int], embedding_dim: int, num_classes: int):
        super().__init__()
        # 4个层级特征的MLP层，将通道数统一
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embedding_dim, 1),
                nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=False) if i > 0 else nn.Identity()
            ) for i, ch in enumerate(in_channels)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, num_classes, 1)
        )

    def forward(self, features):
        # 统一所有层级的分辨率到 1/4 并拼接
        outputs = []
        for i, x in enumerate(features):
            outputs.append(self.linear_layers[i](x))

        x = torch.cat(outputs, dim=1)
        return self.fuse(x)


# --- SegFormer 主体网络 ---
class SegFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dims=[32, 64, 160, 256], heads=[1, 2, 5, 8],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        # Encoder Stages (MiT-B0 风格)
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_channels, dims[0])
        self.block1 = nn.ModuleList([Block(dims[0], heads[0], sr_ratio=sr_ratios[0])])

        self.patch_embed2 = OverlapPatchEmbed(3, 2, dims[0], dims[1])
        self.block2 = nn.ModuleList([Block(dims[1], heads[1], sr_ratio=sr_ratios[1])])

        self.patch_embed3 = OverlapPatchEmbed(3, 2, dims[1], dims[2])
        self.block3 = nn.ModuleList([Block(dims[2], heads[2], sr_ratio=sr_ratios[2])])

        self.patch_embed4 = OverlapPatchEmbed(3, 2, dims[2], dims[3])
        self.block4 = nn.ModuleList([Block(dims[3], heads[3], sr_ratio=sr_ratios[3])])

        # Decoder Head
        self.decoder = SegFormerHead(in_channels=dims, embedding_dim=256, num_classes=num_classes)

    def forward(self, x):
        H_in, W_in = x.shape[2:]

        # Stage 1 (1/4)
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        f1 = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        # Stage 2 (1/8)
        x, H, W = self.patch_embed2(f1)
        for blk in self.block2: x = blk(x, H, W)
        f2 = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        # Stage 3 (1/16)
        x, H, W = self.patch_embed3(f2)
        for blk in self.block3: x = blk(x, H, W)
        f3 = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        # Stage 4 (1/32)
        x, H, W = self.patch_embed4(f3)
        for blk in self.block4: x = blk(x, H, W)
        f4 = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        # Decode
        logits = self.decoder([f1, f2, f3, f4])

        # 最后上采样到原图大小
        return F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)


if __name__ == '__main__':
    # 测试代码 (MiT-B0 配置)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormer(in_channels=3, num_classes=2).to(device)

    x = torch.randn(1, 3, 512, 512).to(device)
    output = model(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")  # 预期: [1, 2, 512, 512]