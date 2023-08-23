
import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
import math
import multiprocessing
from layer.bitflip import int2bin,bin2int
from bitstring import BitArray
import matplotlib.pyplot as plt 
from layer.fi import operation_fi


#relu
def relu(x, bit):


       
    a = torch.zeros_like(x)
    x = torch.max(x, a)

    return x 


class ReLU(nn.Module):
    def __init__(self, bit):
        super(ReLU_fi, self).__init__()

        self.bit = bit

    def forward(self, x):
        
        y = relu(x, self.bit)

        return y 


def relu_fi(x, ber, bit):

       
    a = torch.zeros_like(x)
    x = operation_fi(x, ber, bit)
    x = torch.max(x, a)    
    return x 

class ReLU_fi(nn.Module):
    def __init__(self, ber, bit):
        super(ReLU_fi, self).__init__()

        self.ber = ber
        self.bit = bit
        
    def forward(self, x):
        
        y = relu_fi(x, self.ber, self.bit)        
       
        return y 

#gelu
def gelu(x, bit):

    x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))            
    return x


class GELU(nn.Module):

    def __init__(self, bit):
        super(GELU, self).__init__()

        self.bit = bit        

    def forward(self, x):
        
        y = gelu(x, self.bit)           
  
        return y


def gelu_fi(x, ber, bit):
        

        a = math.sqrt(2 / math.pi)
        
        b = 0.044715 * torch.pow(x, 3)   
        
        b = operation_fi(b, ber, bit)
     
        c = x + b
        
        c = operation_fi(c, ber, bit)     
        
        d = a * c
       
        d = operation_fi(d, ber, bit)
        
        e = 1 + torch.tanh(d)
        
        e = operation_fi(e, ber, bit)
                
        f = 0.5 * x 
        
        f = operation_fi(f, ber, bit)
      
        x = f * e 
        
        x = operation_fi(x, ber, bit)  
        
        return x  

class GELU_fi(nn.Module):

    def __init__(self, ber, bit):
        super(GELU_fi, self).__init__()
        
        self.ber = ber
        self.bit = bit
        
    def forward(self, x):               
        
        y = gelu_fi(x, self.ber, self.bit)          

        return y
        


#softmax
def softmax(x, dim, bit):

        x_exp = x.exp()        
        partition = x_exp.sum(dim, keepdim=True)
        x = x_exp / partition  
        return x    
        
class Softmax(nn.Module):

    def __init__(self, dim, bit):

        super(Softmax, self).__init__()
        self.dim = dim 
        self.bit = bit               
        
    def forward(self, x):

        y = softmax(x, self.dim, self.bit)
        
        return y  

def softmax_fi(x, dim, ber, bit):

        x_exp = x.exp()
        x_exp = operation_fi(x_exp, ber, bit) 
        
        partition = x_exp.sum(dim, keepdim=True)        
        partition = operation_fi(partition, ber, bit) 
        
        x = x_exp / partition               
        x = operation_fi(x, ber, bit) 
        return x 

class Softmax_fi(nn.Module):

    def __init__(self, dim, ber, bit):
        super(Softmax_fi, self).__init__()
        
        self.dim = dim                
        self.ber = ber
        self.bit = bit
                
    def forward(self, x):        
                     
        y = softmax_fi(x, self.dim, self.ber, self.bit)

        return y  


#AvgPool2d

def avgpool2d(x, bit, kernel_size, stride):

    x = F.avg_pool2d(x, kernel_size, stride)
    return x

class AvgPool2d(nn.Module):
    def __init__(self, bit, kernel_size=2, stride=2):
        super(AvgPool2d, self).__init__()
        self.bit = bit
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
 
        y = avgpool2d(x, self.bit, self.kernel_size, self.stride)
      
        return y


def avgpool2d_fi(x, ber, bit, kernel_size, stride):

        x = operation_fi(x, ber, bit)      
        x = F.avg_pool2d(x, kernel_size, stride)
        return x

        
class AvgPool2d_fi(nn.Module):
    def __init__(self, ber, bit, kernel_size=2, stride=2):
        super(AvgPool2d_fi, self).__init__()
        self.ber = ber
        self.bit = bit
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
   
        y = avgpool2d_fi(x, self.ber, self.bit, self.kernel_size, self.stride)
      
        return x


#MaxPool2d


def maxpool2d(x, bit, kernel_size, stride, padding):
    x = F.max_pool2d(x, kernel_size, stride, padding)
    return x

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding = 0):
        super(MaxPool2d, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.bit = bit
        self.padding= padding
                
    def forward(self, x):
    
        y = maxpool2d(x, self.bit, self.kernel_size, self.stride, self.padding)
        return y

def maxpool2d_fi(x, ber, bit, kernel_size, stride, padding):
        x = operation_fi(x, ber, bit)      
        x = F.max_pool2d(x, kernel_size, stride, padding)
        return x

class MaxPool2d_fi(nn.Module):
    def __init__(self, ber, bit, kernel_size, stride, padding = 0):
        super(MaxPool2d_fi, self).__init__()
        self.ber = ber
        self.bit = bit
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding= padding

    def forward(self, x):

        y = maxpool2d_fi(x, self.ber, self.bit, self.kernel_size, self.stride, self.padding)
        return y       




#LayerNorm

class LayerNorm(nn.Module):

    def __init__(self, features, epsilon=1e-6):

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        x = x - mean
        x = self.weight * x
                
        x = x / (std + self.epsilon) + self.bias

              
        return x        


class LayerNorm_fi(nn.Module):

    def __init__(self, ber, bit, features, epsilon=1e-6):
        super(LayerNorm_fi, self).__init__()
        
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
        self.ber = ber
        self.bit = bit
        
    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        x = x - mean
        x = operation_fi(x, self.ber, self.bit) 
        
        x = self.weight * x
        x = operation_fi(x, self.ber, self.bit) 
        
        x = x / (std + self.epsilon)
        x = operation_fi(x, self.ber, self.bit) 
        
        x = x + self.bias        
        x = operation_fi(x, self.ber, self.bit) 
                      
        return x     
        
                        
#BatchNorm2d        
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_batches_tracked = nn.Parameter(torch.tensor(0.))
        self.num_channels = num_features        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        assert b > 1

        if self.training or not self.track_running_stats:
            # All dims except C
            mu = x.mean(dim=(0, 2, 3))
            sigma = x.var(dim=(0, 2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((b * h * w) / ((b * h * w) - 1))
            self.running_mean = self.running_mean * (1 - self.momentum) + mu * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased * self.momentum

        mu = mu.reshape(1, c, 1, 1)
        sigma = sigma.reshape(1, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        param_shape = [1] * len(result.shape)
        param_shape[1] = self.num_channels
        result = result * self.weight.reshape(*param_shape) + self.bias.reshape(*param_shape)        
        
        return result
        
class BatchNorm2d_fi(nn.Module):
    def __init__(self, num_features, ber, bit, momentum=0.1, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()

        self.ber = ber
        self.bit = bit
        
        self.momentum = momentum
        self.eps = eps
        self.num_batches_tracked = nn.Parameter(torch.tensor(0.))
        self.num_channels = num_features        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.track_running_stats = track_running_stats

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))


    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        assert b > 1
        mu = x.mean(dim=(0, 2, 3))
        sigma = x.var(dim=(0, 2, 3), unbiased=False)
        sigma_unbiased = sigma * ((b * h * w) / ((b * h * w) - 1))               
        self.running_mean = self.running_mean * (1 - self.momentum) + mu * self.momentum
        self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased * self.momentum
        mu = mu.reshape(1, c, 1, 1)
        sigma = sigma.reshape(1, c, 1, 1)        
                  
        x = x - mu        
        x = operation_fi(x, self.ber, self.bit)
        
        result = x / torch.sqrt(sigma + self.eps)    
        result = operation_fi(result, self.ber, self.bit)     
        
        param_shape = [1] * len(result.shape)
        param_shape[1] = self.num_channels                 
        result = result * self.weight.reshape(*param_shape)
        result = operation_fi(result, self.ber, self.bit)
        
        
        result = result + self.bias.reshape(*param_shape)            
        result = operation_fi(result, self.ber, self.bit)           
    
        return result

       
  

        





        