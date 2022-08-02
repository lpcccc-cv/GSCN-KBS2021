from model import common_ours
from model import common
import torch.nn as nn
import torch
from model import visualize
import torch.nn.functional as F
def make_model(args, parent=False):
    return TEMP(args)



class TEMP(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(TEMP, self).__init__()

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

        m_body = []
        for i in range(self.n_resblocks):
            # m_body.append(ResidualGroup(conv=conv, n_feat=n_feats, kernel_size=kernel_size))
            m_body.append(common_ours.GFF_resblock(expand_channel, group_number=4))
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.m_body = nn.Sequential(*m_body)

        modules_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.tail = nn.Sequential(*modules_tail)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.GFF = conv(36*n_feats, n_feats, 1)
    def forward(self, x):
        x = self.sub_mean(x)
        SFE = self.head(x)

        ##EDSR+GFF
        temp = SFE
        mid_feature = []
        for i in range(36):
            temp = self.m_body[i](temp)
            mid_feature.append(temp)
        x = self.m_body[36](self.GFF(torch.cat(mid_feature,1)))


         ##EDSR
        # x = self.m_body(SFE)
        x = x+SFE
        pred = self.tail(x)
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

