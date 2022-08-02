from model import common
import torch
import torch.nn as nn
from model import blocks
from torchsummary import summary


def make_model(args, parent=False):
    return OURS(args)


# 空间注意力模块
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
        return sa*x+x

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, sa):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        self.use_sa = sa

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)
        self.sa = SALayer()
    def forward(self, x):
        res = self.LFF(self.convs(x))
        if self.use_sa == True:
            res = self.sa(res)
        return res

class OURS(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(OURS, self).__init__()

        n_color = 3
        self.block_number = args.n_resblocks
        n_feats = args.n_feats
        self.use_sa = args.use_sa
        self.global_learn = args.if_global
        self.mid_supervise = args.mid_supervise

        G = 64
        C = 6
        self.scale = args.scale[0]
        ### 输入
        self.conv_input = conv(n_color, n_feats, 3)
        ### 中间层提取特征
        self.rdab = RDB(growRate0 = n_feats, growRate = G, nConvLayers = C, sa = self.use_sa)
        self.conv_cat = conv((self.block_number+1)*n_feats, n_feats, 1)

        self.tail = nn.Sequential(*[
            nn.UpsamplingNearest2d(scale_factor=self.scale),
            conv(n_feats, n_feats, 3),
            nn.ReLU(),
            conv(n_feats, n_feats, 3),
            nn.ReLU(),
            conv(n_feats, args.n_colors, 3)])
        ### 亚像素上采样输出
        self.res_conv = conv(n_feats, n_feats, 3)
        # m_tail = [
        #     common.Upsampler(conv, self.scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, 3)
        # ]
        # self.tail = nn.Sequential(*m_tail)
        # 中间层监督
        self.weights = nn.Parameter(torch.ones(1, self.block_number) / self.block_number, requires_grad=True)

    def forward(self, x):
        residual = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        inputs = self.conv_input(x)

        mid_feature = []
        RDB = inputs
        # 监督
        if self.mid_supervise == True:
            for i in range(self.block_number):
                RDB = self.rdab(RDB)   ###相当于11个卷基层
                if i == 0:
                    temp = RDB
                else:
                    temp = RDB+inputs
                mid_feature.append(temp)
            for i, value in enumerate(mid_feature):
                UP = self.tail(value)
                if i == 0:
                    pred = torch.zeros_like(UP)
                    pred = pred + UP.mul(self.weights.data[0][i])
                else:
                    pred = pred + UP.mul(self.weights.data[0][i])
        # 无监督
        else:
            # mid_feature.append(inputs)
            for i in range(self.block_number):
                RDB = self.rdab(RDB)   ###相当于11个卷基层
                # mid_feature.append(RDB)
            # res = self.conv_cat(torch.cat(mid_feature, 1))
            pred = self.tail(inputs+RDB)

        if self.global_learn:
            pred = torch.add(pred, residual)
        return pred


