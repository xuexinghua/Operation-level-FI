
import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
import math
import multiprocessing
from layer.quantization import Quant
from layer.fi import operation_fi, operation_mulfi



def split_c_error(c, c_fi, n):
    
    b = c.shape[0]
    num = math.ceil(b/n)

    c_fi_a = c_fi[:n, :,:, 1:-1,...]
    c_a = c[:n, :,:, 1:-1,...]    
    c_err_a = c_fi_a - c_a     
    c_error_y = c_err_a.sum(dim=3)
    
    
    for i in range(1, num-1):
        
        c_i = c[i*n:(i+1)*n, :, :, 1:-1,...]
        c_fi_i = c_fi[i*n:(i+1)*n, :, :, 1:-1,...]
        c_err_i = c_fi_i - c_i   
        c_error_i = c_err_i.sum(dim=3)
        
        c_error_y = torch.cat([c_error_y, c_error_i], 0)        
        
    c_fi_b = c_fi[(num-1)*n:, :,:, 1:-1,...]
    c_b = c[(num-1)*n:, :,:, 1:-1,...]    
    c_err_b = c_fi_b - c_b     
    c_error_b = c_err_b.sum(dim=3)  
    
    c_error = torch.cat([c_error_y, c_error_b], 0)

    return c_error


def diret_conv2d(in_feature, kernel, padding, bits, stride, bias=None):

    quant = Quant(bits)
    kernel = quant(kernel) 
             
    batch = in_feature.size(0)
    in_channel = in_feature.size(1)
    orig_h, orig_w = in_feature.size(2), in_feature.size(3)
    out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
    padding = padding

    stride = stride
    out_rows = ((orig_h + 2*padding - keh) // stride) + 1
    out_cols = ((orig_w + 2*padding - kew) // stride) + 1
    inp_unf = torch.nn.functional.unfold(in_feature, (keh, kew), padding=padding, stride=stride)          
    w = kernel.contiguous().view(kernel.size(0), -1).t()
    x = inp_unf.transpose(1, 2)                    
    x1= x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
    w1 = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)
    p = x1*w1                                 
 
    c = torch.cumsum(p, dim=3)
    y = c[:,:,:, -1,...]
                   
    y = y.transpose(1,2)            
    if bias is None:        
         out_unf = y.transpose(1, 2)
    else:
         out_unf = (y + bias).transpose(1, 2)        
    out = out_unf.contiguous().view(batch, out_channel, out_rows,out_cols)
    
    return out

class conv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, bits, stride, padding, bias=None):
		super(conv2d, self).__init__()		

		self.bits = bits	
		self.stride = stride
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
 
		if bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
			self.register_parameter('bias', None)
      
	def forward(self, x):
		return diret_conv2d(x, self.weight, self.padding, self.bits, self.stride, self.bias)    
   
   


def diret_conv2d_fi(in_feature, kernel, padding, ber, bits, stride, bias=None):
    
    quant = Quant(bits)
    kernel = quant(kernel) 

    batch = in_feature.size(0)
    in_channel = in_feature.size(1)
    orig_h, orig_w = in_feature.size(2), in_feature.size(3)
    out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
    padding = padding

    stride = stride
    out_rows = ((orig_h + 2*padding - keh) // stride) + 1
    out_cols = ((orig_w + 2*padding - kew) // stride) + 1
    inp_unf = torch.nn.functional.unfold(in_feature, (keh, kew), padding=padding, stride=stride)      
      
    w = kernel.contiguous().view(kernel.size(0), -1).t()
    x = inp_unf.transpose(1, 2)                    
    x1= x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
    w1 = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)
    
    p = x1*w1                                 
    
    p = operation_mulfi(p, ber, bits)   
        
    c = torch.cumsum(p, dim=3)        
    c_fi = operation_fi(c, ber, bits)     
    y_sum = c_fi[:,:,:, -1,...]           
    
    #c_error = split_c_error(c, c_fi, 1)   
    c_err = c_fi[:,:,:,1:-1,...] - c[:,:,:,1:-1,...]    
    c_error = c_err.sum(dim=3)         
        
    y = y_sum + c_error	
    y = operation_fi(y, ber, bits)    
        
    #y = (p).sum(3)
                  
    y = y.transpose(1,2)  
              
    if bias is None:        
         out_unf = y.transpose(1, 2)
    else:
         out_unf = (y + bias).transpose(1, 2)        
    out = out_unf.contiguous().view(batch, out_channel, out_rows,out_cols)

    #out = operation_fi(out, ber, bits)  
    return out


class conv2d_fi(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, ber, bits, stride, padding, bias=None):
		super(conv2d_fi, self).__init__()		

		self.ber = ber
		self.bits = bits	
		self.stride = stride
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
 
		if bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
			self.register_parameter('bias', None)
      
	def forward(self, x):
		return diret_conv2d_fi(x, self.weight, self.padding, self.ber, self.bits, self.stride, self.bias) 
   