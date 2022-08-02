from model import common
from model import common_ours
import torch.nn as nn
import torch
from torchstat import stat
from torchsummary import summary
from thop import profile

class OURS(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(OURS, self).__init__()

        self.recursive_number = 5
        n_feats = 64
        kernel_size = 3
        scale = 2
        n_channel = 3

        self.head = conv(n_channel, n_feats, kernel_size)

        self.rdb = common_ours.GFF_resblock()
        # 原始上采样方法
        # self.up = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
        #             conv(n_feats, n_channel, kernel_size))

        ## 新上采样方法
        self.uprec = nn.Sequential(
                                   conv(n_feats, n_feats, 3),
                                   conv(n_feats, scale**2*n_channel, 5),
                                   nn.PixelShuffle(scale),
                                   # conv(n_channel, n_channel, 5)
                                   )

        self.weights = nn.Parameter(torch.ones(1, self.recursive_number)/self.recursive_number, requires_grad=True)


    def forward(self, x):
        x = self.head(x)

        res = x
        mid_feature = []
        for _ in range(self.recursive_number):
            res = self.rdb(res)
            mid_feature.append(res)

        pred = self.uprec(mid_feature[0]) * self.weights.data[0][0]
        for i in range(1, self.recursive_number):
            pred = pred + self.uprec(mid_feature[i]) * self.weights.data[0][i]

        return pred

# model = OURS().cuda()
# summary(model, (3, 5, 5))


model = OURS()
# stat(model, (3, 5, 5))
data = torch.randn(1, 3, 5, 5)
flops, params = profile(model, inputs=(data,))
print('Flops:',flops/1000000, 'Param:', params)





'''
class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, True)  #FENet: staic(bn)+relu+conv1
        self.reconstructor = BNReLUConv(channels, in_channels, True)      #ReconNet: static(bn)+relu+conv 
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )
        #ModuleList can be indexed like a regular Python list, but modules it contains are 
        #properly registered, and will be visible by all Module methods.
        
        
        self.weights = nn.Parameter((torch.ones(1, num_memblock)/num_memblock), requires_grad=True)  
        #output1,...,outputn corresponding w1,...,w2


    #Multi-supervised MemNet architecture
    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        w_sum=self.weights.sum(1)  
        mid_feat=[]   # A lsit contains the output of each memblock
        ys = [out]  #A list contains previous memblock output(long-term memory)  and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  #out is the output of GateUnit  channels=64
            mid_feat.append(out);
        #pred = Variable(torch.zeros(x.shape).type(dtype),requires_grad=False)
        pred = (self.reconstructor(mid_feat[0])+residual)*self.weights.data[0][0]/w_sum
        for i in range(1,len(mid_feat)):
            pred = pred + (self.reconstructor(mid_feat[i])+residual)*self.weights.data[0][i]/w_sum

        return pred

    #Base MemNet architecture
    # def forward(self, x):
    #     residual = x   #input data 1 channel
    #     out = self.feature_extractor(x)
    #     ys = [out]  #A list contains previous memblock output and the output of FENet
    #     for memory_block in self.dense_memory:
    #         out = memory_block(out, ys)
    #     out = self.reconstructor(out)
    #     out = out + residual
    #     
    #     return out


'''