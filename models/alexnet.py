#-*-coding:utf8-*-

import torch
import torch.nn as nn
from layer.conv_layers import conv2d_fi
from layer.winograd_layers import winconv2d_fi, winconv2d
from layer.fft_layer import FFTConv2d_fi, FFTConv2d

tiles = 2


class fi_AlexNet(nn.Module):
    def __init__(self, ber, bit, num_classes=10):
        super(fi_AlexNet, self).__init__()

        self.ber = ber
        self.bit = bit
        
        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # shape is 55 x 55 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1), # shape is 27 x 27 x 64
            

            conv2d_fi(64, 192,  3, self.ber, self.bit, stride=1, padding=1,bias=True), # shape is 27 x 27 x 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1), # shape is 13 x 13 x 192

            conv2d_fi(192, 384,  3, self.ber, self.bit, stride=1, padding=1,bias=True), # shape is 13 x 13 x 384
            nn.ReLU(),

            conv2d_fi(384, 256,  3, self.ber, self.bit, stride=1, padding=1,bias=True), # shape is 13 x 13 x 256
            nn.ReLU(),

            conv2d_fi(256, 256,  3, self.ber, self.bit, stride=1, padding=1,bias=True), # shape is 13 x 13 x 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    