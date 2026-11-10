# models/attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.sa(x * self.ca(x))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(F_l)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return self.cbam(x * psi)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(AttentionUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_in = in_channels
        for feature in features[:-1]:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        self.bottleneck = DoubleConv(features[-2], features[-1])

        for i in range(len(features) - 2, -1, -1):
            feature = features[i]
            F_int = max(1, feature // 2)
            self.ups.append(
                nn.ModuleDict({
                    'attention': AttentionGate(F_g=feature, F_l=feature, F_int=F_int),
                    'upconv': nn.ConvTranspose2d(features[i+1], feature, kernel_size=2, stride=2),
                    'conv': DoubleConv(feature * 2, feature)
                })
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx, up_dict in enumerate(self.ups):
            x = up_dict['upconv'](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x_attn = up_dict['attention'](g=x, x=skip_connection)
            x = torch.cat([x_attn, x], dim=1)
            x = up_dict['conv'](x)

        return self.final_conv(x)
