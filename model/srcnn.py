import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common

def make_model(args, parent=False):
    return SRCNN(args)


class SRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRCNN, self).__init__()
        self.scale = args.scale[0]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out




