
import torch
import torch.nn as nn
from layer.conv_layers import conv2d_fi
from layer.winograd_layers import winconv2d_fi, winconv2d
from layer.activate_layer import BatchNorm2d_fi, ReLU_fi, MaxPool2d_fi, AvgPool2d_fi, ReLU
from layer.fft_layer import FFTConv2d_fi, FFTConv2d

tiles = 2


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, bit):
        super(VGG, self).__init__()
        
        self.bit = bit              
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1,bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]
                                                      
                in_channels = x                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class winVGG(nn.Module):
    def __init__(self, vgg_name, bit):
        super(winVGG, self).__init__()

        self.bit = bit
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [winconv2d(in_channels, x, 3, tiles, self.bit, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]   
                                                 
                in_channels = x                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)






class fftVGG(nn.Module):
    def __init__(self, vgg_name, bit):
        super(fftVGG, self).__init__()

        self.bit = bit
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [FFTConv2d(in_channels, x, 3, self.bit, padding=1, stride=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]   
                                               
                in_channels = x                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class fi_fftVGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(fi_fftVGG, self).__init__()

        self.ber = ber
        self.bit = bit
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [FFTConv2d_fi(in_channels, x, 3, self.ber, self.bit, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]   
                                               
                in_channels = x                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class fi_VGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(fi_VGG, self).__init__()
        
        self.ber = ber
        self.bit = bit
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [conv2d_fi(in_channels, x, 3, self.ber, self.bit, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]  
                in_channels = x     
                           
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class fi_winVGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(fi_winVGG, self).__init__()

        self.ber = ber
        self.bit = bit
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [winconv2d_fi(in_channels, x, 3, tiles, self.ber, self.bit, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU()]                     
                in_channels = x   
                             
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)







