from model import common_ours
from model import common
import torch.nn as nn
import torch
import torch.nn.functional as F
def make_model(args, parent=False):
    return GFFNET_M(args)




class GFFNET_M(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(GFFNET_M, self).__init__()

        self.n_resblocks = args.n_resblocks  # 总残差块个数
        self.group_number = 4  ## 分组卷积个数
        expand_channel = args.expend_channel

        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        self.global_learn = args.if_global

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*m_head)

        self.m_body = nn.ModuleList()
        for i in range(self.n_resblocks):
            self.m_body.append(common_ours.GFF_resblock(expand_channel,  group_number=self.group_number))
        self.conv_tail = conv(n_feats, n_feats, 3)

        m_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        SFE = self.head(x)
        x = SFE

        for i in range(self.n_resblocks):
            x = self.m_body[i](x)
        res = self.conv_tail(x)+SFE
        pred = self.tail(res)
        pred = self.add_mean(pred)
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

