'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantizer import *


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,is_pwc=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        if is_pwc== 1:
            self.conv2 = Quant_Conv2d(in_planes, out_planes, kernel_size=1, stride = 1, padding = 0, bias = False)
        else :
	        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2,0), 128, (256,2,0), 256, (512,2,0), 512, 512, 512, 512, 512, (1024,2,0), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.cnt = 0

    def _make_layers(self, in_planes):
        layers = []
        cnt = 0
        for x in self.cfg:
            
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
#            is_pwc = 0 if isinstance(x,int) else x[2]
            if cnt == 0:
                is_pwc=1
            else:
                is_pwc=0
            cnt = 1
            layers.append(Block(in_planes, out_planes, stride,is_pwc))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
