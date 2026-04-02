
from transformers import CLIPTextModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, text_dim=512):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.text_proj = nn.Linear(text_dim, F_int)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x, t_embed):

        g1 = self.W_g(g)

        if g1.size()[2:] != x.size()[2:]:
            g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.W_l(x)

        t_feat = self.text_proj(t_embed).unsqueeze(-1).unsqueeze(-1)

        combined = self.relu(g1 + x1 + t_feat)

        attention_map = self.psi(combined)

        out = x * (1 + attention_map)

        return out, attention_map


class IGRCBlock(nn.Module):
    """
    IGRC Block: Infrared Granularity Recalibration Block
    Designed for infrared small-object segmentation.
    Residual convolution + channel recalibration.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super(IGRCBlock, self).__init__()

        # Residual convolution brancha
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Channel recalibration (SE)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        b, c, _, _ = out.size()  # ✅ 改这里

        y = self.avg_pool(out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        out = out * y.expand_as(out)

        out = out + self.shortcut(x)
        return self.act(out)


class TACAM(nn.Module): #Thermal Anisotropic Context Aggregation Module
    def __init__(self, in_channels, out_channels):
        super(TACAM, self).__init__()

        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
                                   nn.SiLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.SiLU())


        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_strip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.fuse = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        h, w = x.size()[2:]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)

        xh = self.pool_h(x).expand(-1, -1, h, w)
        xw = self.pool_w(x).expand(-1, -1, h, w)
        x3 = self.conv_strip(xh + xw)

        out = self.fuse(torch.cat([x1, x2, x3], dim=1))
        return self.silu(self.bn(out))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_l = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g 是来自深层的特征，x 是来自浅层的跳跃连接
        g1 = self.W_g(g)
        if g1.size()[2:] != x.size()[2:]:
            g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)



class IRSegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_c=32, deep_supervision=False):
        super(IRSegNet, self).__init__()
        self.deep_supervision = deep_supervision

        self.text_encoder = CLIPTextModel.from_pretrained(r"D:\zyl\IRSeg\irseg\clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.inc = IGRCBlock(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), IGRCBlock(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), IGRCBlock(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), IGRCBlock(base_c * 4, base_c * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), IGRCBlock(base_c * 8, base_c * 16))


        self.bridge = TACAM(base_c * 16, base_c * 16)


        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag1 = MultiModalAttentionGate(F_g=base_c * 16, F_l=base_c * 8, F_int=base_c * 8)
        self.conv1 = IGRCBlock(base_c * 24, base_c * 8)

        # Layer 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag2 = MultiModalAttentionGate(F_g=base_c * 8, F_l=base_c * 4, F_int=base_c * 4)
        self.conv2 = IGRCBlock(base_c * 12, base_c * 4)

        # Layer 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag3 = MultiModalAttentionGate(F_g=base_c * 4, F_l=base_c * 2, F_int=base_c * 2)
        self.conv3 = IGRCBlock(base_c * 6, base_c * 2)

        # Layer 4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag4 = MultiModalAttentionGate(F_g=base_c * 2, F_l=base_c, F_int=base_c)
        self.conv4 = IGRCBlock(base_c * 3, base_c)

        # Final Output
        self.outc = nn.Conv2d(base_c, num_classes, 1)

        if self.deep_supervision:
            self.ds3 = nn.Conv2d(base_c * 2, num_classes, 1)
            self.ds2 = nn.Conv2d(base_c * 4, num_classes, 1)

    def forward(self, x, input_ids, attention_mask):

        text_outputs = self.text_encoder(input_ids=input_ids,
                                         attention_mask=attention_mask)
        t_embed = text_outputs.pooler_output  # [B, 512]

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bridge
        x5 = self.bridge(x5)

        # ---------------- Decoder ----------------
        attention_maps = []

        # Layer 1
        u1 = self.up1(x5)
        a1, att1 = self.ag1(x5, x4, t_embed)
        attention_maps.append(att1)
        u1 = self.conv1(torch.cat([u1, a1], dim=1))

        # Layer 2
        u2 = self.up2(u1)
        a2, att2 = self.ag2(u1, x3, t_embed)
        attention_maps.append(att2)
        u2 = self.conv2(torch.cat([u2, a2], dim=1))

        # Layer 3
        u3 = self.up3(u2)
        a3, att3 = self.ag3(u2, x2, t_embed)
        attention_maps.append(att3)
        u3 = self.conv3(torch.cat([u3, a3], dim=1))

        # Layer 4
        u4 = self.up4(u3)
        a4, att4 = self.ag4(u3, x1, t_embed)
        attention_maps.append(att4)
        u4 = self.conv4(torch.cat([u4, a4], dim=1))

        logits = self.outc(u4)

        if self.training:
            if self.deep_supervision:
                out3 = F.interpolate(self.ds3(u3), size=x.size()[2:], mode='bilinear', align_corners=True)
                out2 = F.interpolate(self.ds2(u2), size=x.size()[2:], mode='bilinear', align_corners=True)
                return logits, out3, out2, attention_maps

            return logits, attention_maps

        return logits, attention_maps
