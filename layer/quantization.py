import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math

# Code adapted from: https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py
#https://github.com/jafermarq/WinogradAwareNets/blob/master/src/quantization.py

def Quantize(input, num_bits=8, min_value=None, max_value=None, stochastic=False, out_half=False):

        output = input.clone()
        
        num_bits = int(num_bits)

        qmin = -1.0 * (2**num_bits)/2
        qmax = -qmin - 1

        # compute qparams --> scale and zero_point
        max_val, min_val = float(max_value), float(min_value)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)

        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            max_range = max(-min_val, max_val) # largest mag(value)
            scale = max_range / ((qmax - qmin) / 2)
            scale = max(scale, 1e-8)
            zero_point = 0.0 # this true for symmetric quantization

        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output.div_(scale).add_(zero_point)
        output.round_().clamp_(qmin, qmax)  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize
        

        if out_half and num_bits <= 16:
            output = output.half()

        return output


class Quant(nn.Module):

    def __init__(self, num_bits=8,momentum=0.0078):
        super(Quant, self).__init__()
        self.min_val= 0
        self.max_val=0
        self.momentum = momentum
        self.num_bits = num_bits


    def forward(self, input):

            min_val = self.min_val
            max_val = self.max_val
            #print('min_val1',min_val)
            #print('max_val1',max_val)

            if min_val == max_val: # we'll reach here if never obtained min/max of input
                min_val = input.detach().min() 
                max_val = input.detach().max() 
            else:
                #print('min_val2',input.detach().min())
                #print('max_val2',input.detach().max())
                
                # equivalent to --> min_val = min_val(1-self.momentum) + self.momentum * torch.min(input)
                min_val = min_val + self.momentum * (input.detach().min()  - min_val)
                max_val = max_val + self.momentum * (input.detach().max()  - max_val)
                #print('min_val3',min_val)
                #print('max_val3',max_val)
            self.min_val = torch.tensor(self.min_val)
            self.max_val = torch.tensor(self.max_val)    
                    
            self.min_val = min_val
            self.max_val = max_val

            return Quantize(input, self.num_bits, self.min_val, self.max_val)
