import os
import math
import torch
import torch.nn as nn
import torchvision.models
from layer.conv_layers import conv2d_fi
from layer.winograd_layers import winconv2d_fi, winconv2d
from layer.fft_layer import FFTConv2d_fi, FFTConv2d
tiles = 2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ber, bit, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class fi_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ber, bit, inplanes, planes, stride=1, downsample=None):
        super(fi_BasicBlock, self).__init__()
        self.conv1 = conv2d_fi(inplanes, planes, 3, ber, bit, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2d_fi(planes, planes, 3, ber, bit, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class fi_win_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ber, bit, inplanes, planes, stride=1, downsample=None):
        super(fi_win_BasicBlock, self).__init__()
        
        self.conv1 = winconv2d_fi(inplanes, planes, 3, tiles, ber, bit, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = winconv2d_fi(planes, planes, 3, tiles, ber, bit, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, ber, bit, block_1, block_2, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.ber = ber
        self.bit = bit           
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_1, block_2, 64, layers[0])
        self.layer2 = self._make_layer(block_1, block_2, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_1, block_2, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_1, block_2, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block_1.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block_1, block_2, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_1.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_1.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_1.expansion),
            )

        layers = []
        layers.append(block_2(self.ber, self.bit, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_1.expansion
        for i in range(1, blocks):
            layers.append(block_1(self.ber, self.bit, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def fi_resnet18(ber, bit, pretrained=False, **kwargs):

    model = ResNet(ber, bit, fi_BasicBlock, BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def fi_win_resnet18(ber, bit, pretrained=False, **kwargs):

    model = ResNet(ber, bit, fi_win_BasicBlock, BasicBlock, [2, 2, 2, 2], **kwargs)

    return model
