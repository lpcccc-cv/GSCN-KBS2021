## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

def make_model(args, parent=False):
    return IRAN(args)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # 平均池化为64维向量
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),    # 减少为4维
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),     # 扩充为64维
                nn.Sigmoid()    # 转化为权重
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y    # 通道加权
class SALayer(nn.Module):
    def __init__(self, kernel_size=5):
        super(SALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.pool_cov1 = nn.Conv2d(2, 2, kernel_size, padding=kernel_size//2, bias=True)
        self.pool_cov2 = nn.Conv2d(2, 1, 1, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        max_pool = torch.max(x, 1,keepdim=True)[0]
        avg_pool = torch.mean(x, 1, keepdim=True)
        pool_layer = torch.cat((max_pool, avg_pool), 1)

        y = self.pool_cov1(pool_layer)
        y = self.relu(y)
        y = self.pool_cov2(y)
        spacial_attention = self.sigmoid(y)
        res = x * spacial_attention

        return res

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        self.relu = act
        self.ca_layer = CALayer(n_feat, reduction)  # channel attention layer
        self.sa_layer = SALayer()
        self.cov_x1 = conv(n_feat, n_feat, kernel_size=1, bias=bias)
        self.cov_x3_1 = conv(n_feat, n_feat, kernel_size=3, bias=bias)
        self.cov_x3_2 = conv(n_feat, n_feat, kernel_size=3, bias=bias)
        self.cov_concat = conv(n_feat * 3, n_feat, kernel_size=1, bias=bias)
        self.cov_x5 = conv(n_feat, n_feat, kernel_size=5, bias=bias)

    def forward(self, x):
        res_x1 = self.cov_x1(x)
        res_x3 = self.cov_x3_1(self.relu(res_x1))
        res_x5 = self.cov_x3_2(self.relu(res_x3))

        concat = torch.cat((res_x1, res_x3, res_x5), 1)
        res = self.cov_concat(concat)
        res_ca = self.ca_layer(res)
        res_sa = self.sa_layer(res)

        return res+res_sa+x


## Residual Channel Attention Network (RCAN)
class IRAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(IRAN, self).__init__()

        n_colors = 3  # 颜色通道个数
        n_resblocks = 16  # 残差块个数
        n_feats = 64   # feature map个数
        kernel_size = 3
        reduction = 16   # number of feature maps reduction
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            RCAB(
                conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.upsample = common.Upsampler(conv, self.scale, n_feats, act=False)
        self.cov = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.head(x)

        res = self.body(x)

        res += x
        res = self.upsample(res)

        x = self.cov(res)
        x = x + bicubic
        return x 


# model = RCAN().cuda()
# summary(model, (1, 30, 30))

