import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) module to suppress background clutter (e.g. rocks, shadows)
    and focus on glacier margins.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        psi_conv = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(psi_conv.bias, 1.0)
        
        self.psi = nn.Sequential(
            psi_conv,
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from coarser scale, x: skip connection feature map
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        net = self.relu(g1 + x1)
        att = self.psi(net)
        return x * att

class AttentionUNet(nn.Module):
    """
    8-Channel Attention U-Net Architecture for Glacier Segmentation.
    Input tensor shape: (B, 8, H, W)
    Output tensor shape: (B, 1, H, W) raw logits
    """
    def __init__(self, in_channels=8, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Attention Gates for Skip Connections
        self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.ag4 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # Output Convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Attention Gates
        x_up1 = self.up1(x5)
        x4_att = self.ag1(g=x_up1, x=x4)
        x_d1 = self.conv_up1(torch.cat([x_up1, x4_att], dim=1))

        x_up2 = self.up2(x_d1)
        x3_att = self.ag2(g=x_up2, x=x3)
        x_d2 = self.conv_up2(torch.cat([x_up2, x3_att], dim=1))

        x_up3 = self.up3(x_d2)
        x2_att = self.ag3(g=x_up3, x=x2)
        x_d3 = self.conv_up3(torch.cat([x_up3, x2_att], dim=1))

        x_up4 = self.up4(x_d3)
        x1_att = self.ag4(g=x_up4, x=x1)
        x_d4 = self.conv_up4(torch.cat([x_up4, x1_att], dim=1))

        logits = self.outc(x_d4)
        return logits
