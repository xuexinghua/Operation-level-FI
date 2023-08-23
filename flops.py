from torchvision.models import vgg19
from thop import profile
import torch
model = vgg19()
print(model)
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
print(params)
