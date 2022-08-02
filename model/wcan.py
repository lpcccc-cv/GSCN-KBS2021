## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
def make_model(args, parent=False):
    return WCAN(args)

# Channel Attention (CA) Layer
class CALayer_avg(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_avg, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # 平均池化为64维向量
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),    # 减少为4维
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),     # 扩充为64维
                # nn.Sigmoid()    # 转化为权重
        )

    def forward(self, x):
        y = self.avg_pool(x)
        # print(y)
        y = self.conv_du(y)
        # print(y)
        return y

class CALayer_max(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_max, self).__init__()
        # global average pooling: feature --> point
        self.max_pool = nn.AdaptiveMaxPool2d(1)   # 平均池化为64维向量
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),    # 减少为4维
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),     # 扩充为64维
                # nn.Sigmoid()    # 转化为权重
        )

    def forward(self, x):
        y = self.max_pool(x)
        # print(y)
        y = self.conv_du(y)
        # print(y)
        return y




## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        self.relu = act
        self.ca_layer_avg = CALayer_avg(n_feat, reduction)  # channel attention layer
        self.ca_layer_max = CALayer_max(n_feat, reduction)  # channel attention layer

        self.cov_x3_1 = conv(n_feat, n_feat, kernel_size=3, bias=bias)
        self.cov_x3_2 = conv(n_feat, n_feat, kernel_size=3, bias=bias)

        self.cov_concat = conv(n_feat * 2, n_feat, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res_x3 = self.cov_x3_1(x)
        res_avg = self.ca_layer_avg(res_x3)
        res_max = self.ca_layer_max(res_x3)
        attention = self.sigmoid(res_avg+res_max)

        res = res_x3*attention
        res = res_x3 + res
        res_x3 = self.relu(res)
        res_x5 = self.cov_x3_2(res_x3)

        concat = torch.cat((res_x5, x), 1)
        res = self.cov_concat(concat)

        res = x + res
        return res


## Residual Channel Attention Network (RCAN)
class WCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(WCAN, self).__init__()

        n_colors = 3  # 颜色通道个数
        n_resblocks = args.n_resblocks  # 残差块个数
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
        # self.tail = nn.Sequential(*modules_tail)
        self.upsample = common.Upsampler(conv, self.scale, n_feats, act=False)
        self.cov = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):

        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.head(x)

        res = self.body(x)
        res = self.upsample(res)

        x = self.cov(res)
        x = x + bicubic

        return x 

#
# model = RCAN().cuda()
# summary(model, (1, 300, 300))

