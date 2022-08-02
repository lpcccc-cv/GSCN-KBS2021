from model import common_ours
from model import common
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return GSCN(args)


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.group_number = 4
        self.n_resblocks = n_resblocks
        modules_body = [common_ours.GFF_resblock(256,  group_number=self.group_number) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)
        self.LFF = conv((n_resblocks)*n_feat, n_feat, 1)

    def forward(self, x):
        temp = x
        mid_feature = []
        for i in range(self.n_resblocks):
            temp = self.body[i](temp)
            mid_feature.append(temp)
        # 原模型
        res = self.LFF(torch.cat(mid_feature, 1))
        # 去掉LFF
        # res = temp

        res = res + x
        return res


class GSCN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(GSCN, self).__init__()

        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, n_resblocks=n_resblocks) \
            for _ in range(self.n_resgroups)]
        self.GFF = conv((self.n_resgroups)*n_feats, n_feats, 3)
        self.GFF_conv = conv(n_feats, n_feats, 3)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        temp = x
        mid_feature = []
        for i in range(self.n_resgroups):
            temp = self.body[i](temp)
            mid_feature.append(temp)
        #GFF
        res = self.GFF_conv(self.GFF(torch.cat(mid_feature, 1)))
        # 去掉GFF
        # res = temp

        res = res+x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))