import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import visualize



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y*x

class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()

        self.pool_cov1 = nn.Conv2d(2, 2, 5, padding=2, bias=True)
        self.pool_cov2 = nn.Conv2d(2, 1, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        max_pool = torch.max(x, 1, keepdim=True)[0]
        avg_pool = torch.mean(x, 1, keepdim=True)
        pool_layer = torch.cat((avg_pool, max_pool), 1)

        y1 = self.relu(self.pool_cov1(pool_layer))
        sa = self.sigmoid(self.pool_cov2(y1))
        # a = sa*x + x
        # visualize.draw_features(a.cpu().numpy(), dpi=250)
        # EMA
        return sa*x + x
        ## SA
        # return sa * x
        ## no_SA
        # return x


class GFF_layer(nn.Module):
    def __init__(self, channel):
        super(GFF_layer, self).__init__()

        group = int(channel/4)
        self.group = group
        self.conv1 = nn.Conv2d(channel, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(group, group, 3, 1, 1)
        self.conv32_1 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.conv32_2 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.conv32_3 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = x
        group = self.group
        x1 = conv1[:, 0:group, :, :]
        x2 = conv1[:, group:group*2, :, :]
        x3 = conv1[:, group*2:group*3, :, :]
        x4 = conv1[:, group*3:group*4, :, :]

        y1 = self.relu(self.conv3(x1))

        in2 = torch.cat((x2, y1), 1)
        y2 = self.relu(self.conv32_1(in2))

        in3 = torch.cat((x3, y2), 1)
        y3 = self.relu(self.conv32_2(in3))

        in4 = torch.cat((x4, y3), 1)
        y4 = self.relu(self.conv32_3(in4))

        y = torch.cat((y1, y2, y3, y4), 1)

        conv2 = self.conv1(y)

        return conv2

class GFF_resblock(nn.Module):
    def __init__(self, channel=128, group_number = 4):
        super(GFF_resblock, self).__init__()
        self.conv_expand = nn.Conv2d(64, channel, 1, 1, 0)

        self.sa = SALayer()
        self.ca = CALayer(64)

        if group_number == 4:
            self.gff_layer = GFF_layer(channel=channel)
        else:
            self.gff_layer = GFF_layer8(channel=channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        gff1 = self.conv_expand(x)
        gff2 = self.gff_layer(gff1)
        # SA
        # gff2 = self.sa(gff2)
        # out = x + gff2

        # CA
        # out = self.ca(gff2)+x

        #CA + SA
        # gff2 = self.sa(self.ca(gff2))
        # out = x + gff2

        # no attention
        out = gff2+x

        return out



class GFF_layer8(nn.Module):
    def __init__(self, channel):
        super(GFF_layer8, self).__init__()

        group = int(channel/8)
        self.group = group
        self.conv1 = nn.Conv2d(channel, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(group, group, 3, 1, 1)
        self.conv32_1 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.conv32_2 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.conv32_3 = nn.Conv2d(2*group, group, 3, 1, 1)
        self.conv32_4 = nn.Conv2d(2 * group, group, 3, 1, 1)
        self.conv32_5 = nn.Conv2d(2 * group, group, 3, 1, 1)
        self.conv32_6 = nn.Conv2d(2 * group, group, 3, 1, 1)
        self.conv32_7 = nn.Conv2d(2 * group, group, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = x
        group = self.group
        x1 = conv1[:, 0:group, :, :]
        x2 = conv1[:, group:group * 2, :, :]
        x3 = conv1[:, group * 2:group* 3, :, :]
        x4 = conv1[:, group * 3:group* 4, :, :]
        x5 = conv1[:, group * 4:group * 5, :, :]
        x6 = conv1[:, group * 5:group * 6, :, :]
        x7 = conv1[:, group * 6:group * 7, :, :]
        x8 = conv1[:, group * 7:group * 8, :, :]


        y1 = self.relu(self.conv3(x1))

        in2 = torch.cat((x2, y1), 1)
        y2 = self.relu(self.conv32_1(in2))

        in3 = torch.cat((x3, y2), 1)
        y3 = self.relu(self.conv32_2(in3))

        in4 = torch.cat((x4, y3), 1)
        y4 = self.relu(self.conv32_3(in4))

        in5 = torch.cat((x5, y4), 1)
        y5 = self.relu(self.conv32_4(in5))

        in6 = torch.cat((x6, y5), 1)
        y6 = self.relu(self.conv32_5(in6))

        in7 = torch.cat((x7, y6), 1)
        y7 = self.relu(self.conv32_6(in7))

        in8 = torch.cat((x8, y7), 1)
        y8 = self.relu(self.conv32_7(in8))

        y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 1)

        conv2 = self.conv1(y)

        return conv2

class RDB_L4(nn.Module):
    def __init__(self):
        super(RDB_L4, self).__init__()

        self.compress = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv1 = nn.Conv2d(32+64, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(64+64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(96+64, 64, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        res1 = self.relu(self.compress(x))
        res2 = self.relu(self.conv1(torch.cat((x, res1), 1)))
        res3 = self.relu(self.conv2(torch.cat((x, res1, res2), 1)))
        res4 = self.conv3(torch.cat((x, res1, res2, res3), 1))

        return res4+x


class GFF_resblock_no_sc(nn.Module):
    def __init__(self, channel=128, group_number = 4):
        super(GFF_resblock_no_sc, self).__init__()
        self.conv_expand = nn.Conv2d(64, channel, 1, 1, 0)
        self.conv_group = nn.Conv2d(128, 128, 3, 1, 1, groups=4)
        self.conv_compress = nn.Conv2d(128, 64, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        gff1 = self.conv_group(self.conv_expand(x))
        gff2 = self.conv_compress(self.relu(gff1))
        out = x+gff2

        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

