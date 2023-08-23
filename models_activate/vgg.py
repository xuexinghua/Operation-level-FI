
import torch
import torch.nn as nn
from layer.activate_layer import BatchNorm2d_fi, ReLU_fi, MaxPool2d_fi, AvgPool2d_fi

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class BatchNorm2d_fi_VGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(BatchNorm2d_fi_VGG, self).__init__()
        
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
                layers += [nn.Conv2d(in_channels, x, 3, stride=1, padding=1, bias=True),
                           BatchNorm2d_fi(x, self.ber, self.bit),
                           nn.ReLU()]    
                in_channels = x                             
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class ReLU_fi_VGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(ReLU_fi_VGG, self).__init__()        
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
                layers += [nn.Conv2d(in_channels, x, 3, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           ReLU_fi(self.ber, self.bit)]    
                in_channels = x                                
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)        
        
class MaxPool2d_fi_VGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(MaxPool2d_fi_VGG, self).__init__()
        
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
                layers += [MaxPool2d_fi(self.ber, self.bit, kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, 3, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]    
                in_channels = x                               
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
              
class AvgPool2d_fi_VGG(nn.Module):
    def __init__(self, vgg_name, ber, bit):
        super(AvgPool2d_fi_VGG, self).__init__()
        
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
                layers += [nn.Conv2d(in_channels, x, 3, stride=1, padding=1, bias=True),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]    
                in_channels = x                                
        layers += [AvgPool2d_fi(self.ber, self.bit, kernel_size=1, stride=1)]
        return nn.Sequential(*layers)                               
