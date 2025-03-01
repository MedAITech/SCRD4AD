import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import wide_resnet50_2
from models.de_resnet import de_wide_resnet50_2



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class RdadAttenMax(nn.Module):
    def __init__(
            self,
    ) -> None:
        super(RdadAttenMax, self).__init__()
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        self.decoder = de_wide_resnet50_2(pretrained=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(2048, 3)
        

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            en = self.encoder(x)
        bn = self.bn(en)
        de = self.decoder(bn)
        po = self.pool(bn)
        para = torch.softmax(self.mlp(po.view(po.shape[0], -1)), dim=-1)
        return en, de, para


class RdadAtten(nn.Module):
    def __init__(
            self,
    ) -> None:
        super(RdadAtten, self).__init__()
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        self.decoder = de_wide_resnet50_2(pretrained=False)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Linear(2048, 3)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            en = self.encoder(x)
        bn = self.bn(en)
        de = self.decoder(bn)
        po = self.pool(bn)
        para = torch.softmax(self.mlp(po.view(po.shape[0], -1)), dim=-1)
        return en, de, para  

