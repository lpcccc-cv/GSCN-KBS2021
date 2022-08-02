from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


def make_model(args, parent=False):
    return VDSROUR(args)

class VDSROUR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSROUR, self).__init__()

        n_resblocks = 20
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, args.n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        res = self.body(x)
        res += x
        # x = self.add_mean(res)

        return res

