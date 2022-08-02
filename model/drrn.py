import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from model import common


def make_model(args, parent=False):
    return DRRN(args)

class DRRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.scale = args.scale
        self.n_resblock = args.n_resblocks
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=self.scale[0], mode='bicubic', align_corners=False)
        inputs = self.input(self.relu(residual))
        out = inputs
        for _ in range(self.n_resblock):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out


