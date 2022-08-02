from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

def make_model(args, parent=False):
    return LGCNET(args)


class LGCNET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LGCNET, self).__init__()

        n_feats = 32
        kernel_size = 3
        self.scale = args.scale[0]
        n_colors = 3

        self.relu = nn.ReLU(True)

        # define head module
        self.cov1 = conv(n_colors, n_feats, kernel_size)
        self.cov2= conv(n_feats, n_feats, kernel_size)
        self.cov3 = conv(n_feats, n_feats, kernel_size)
        self.cov4 = conv(n_feats, n_feats, kernel_size)
        self.cov5 = conv(n_feats, n_feats, kernel_size)
        self.cov6 = conv(n_feats*3, 64, 5)
        self.cov7 = conv(64, n_colors, kernel_size)

    def gelu(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.44715*torch.pow(x, 3))))

    def forward(self, x):

        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.cov1(bicubic)
        x = self.relu(x)
        x = self.cov2(x)
        x = self.relu(x)
        cov3 = self.cov3(x)
        cov4 = self.cov4(cov3)
        cov5 = self.cov5(cov4)
        cat = torch.cat((cov3, cov4, cov5), 1)
        x = self.relu(cat)
        x = self.cov6(cat)
        x = self.relu(x)
        x = self.cov7(x)
        x = x + bicubic



        return x 



