from model import common_ours
from model import common
import torch.nn as nn
import torch
import torch.nn.functional as F
def make_model(args, parent=False):
    return DLINET(args)



class DLINET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DLINET, self).__init__()

        self.n_resblocks = args.n_resblocks  # 总残差块个数

        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]

        self.up_module = args.up_module
        self.mid_supervise = args.mid_supervise
        self.global_learn = args.if_global

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*m_head)

        # define body module
        self.m_body = nn.ModuleList()
        for i in range(self.n_resblocks):
            self.m_body.append(
                common_ours.DLI_ResBlock(n_feats)
            )

        self.fuse_conv = conv(16*self.n_resblocks+n_feats, n_feats, 3)

        # 上采样
        self.conv_tail = conv(n_feats, n_feats, kernel_size)
        ## espcn上采样
        m_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        SFE = self.head(x)
        x = SFE
        mid_feature = []
        # 无中间层监督
        for i in range(self.n_resblocks):
            x, mid = self.m_body[i](x)
            mid_feature.append(mid)
        mid_feature.append(x)
        # 蒸馏提取块
        x = x+self.fuse_conv(self.relu(torch.cat(mid_feature, 1)))


        # 上采样
        res = self.conv_tail(x) + SFE
        pred = self.tail(res)

        # 输出
        if self.global_learn == True:
            bicubic = F.interpolate(input, scale_factor=self.scale, mode='bicubic', align_corners=False)
            pred = pred+bicubic
        return  pred

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

