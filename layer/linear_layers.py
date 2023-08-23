
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
    
def quan_linear(x, w, bits, bias=None):      

        quant = Quant(bits)
        w = quant(w)         
        w = w.t()

        x1= x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
        w1 = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)     
        p=x1*w1            
              
        c = torch.cumsum(p, dim=3)
        y = c[:,:,:, -1,...]        

        y = y.transpose(1,2)             
        if bias is not None:  
           y += bias.unsqueeze(0).expand_as(y)      

        return y
                      
class quan_Linear(nn.Module):

	def __init__(self, in_features, out_features, bits, bias=None):
		super(quan_Linear, self).__init__()		
		self.bits = bits
		self.in_features = in_features
		self.out_features = out_features
		self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		if bias:
						self.bias = torch.nn.Parameter(torch.Tensor(out_features))            
		else:
						self.register_parameter('bias', None)
	def forward(self, x):
		return quan_linear(x, self.weight, self.bits, self.bias)  


def quan_linear_fi(x, w, ber, bits, bias=None):      
        
        quant = Quant(bits)
        w = quant(w) 
        w = w.t()
        
        x1= x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
        w1 = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)
     
        p=x1*w1
        
        p = operation_mulfi(p, ber, bits)
                                       
        c = torch.cumsum(p, dim=3)
               
        c_fi = operation_fi(c, ber, bits)        
              
        y_sum = c_fi[:,:,:, -1,...]        
         
        c_err = c_fi[:,:,:, 1:-1,...] - c[:,:,:, 1:-1,...]
        c_error = c_err.sum(dim=-1)        
        
        y = y_sum + c_error
                       
        y = operation_fi(y, ber, bits)    
      
        y = y.transpose(1,2)   
  
                  
        if bias is not None:  
           y += bias.unsqueeze(0).expand_as(y)      
               
        return y


                      
class quan_Linear_fi(nn.Module):

	def __init__(self, in_features, out_features, ber, bits, bias=None):
		super(quan_Linear_fi, self).__init__()		
		self.bits = bits
		self.ber = ber    
		self.in_features = in_features
		self.out_features = out_features
		self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		if bias:
						self.bias = torch.nn.Parameter(torch.Tensor(out_features))            
		else:
						self.register_parameter('bias', None)
	def forward(self, x):
		return quan_linear_fi(x, self.weight, self.ber, self.bits, self.bias)



    
def GEMM(a, b):
    b = b.transpose(2,3)
    a1 = a.unsqueeze(3).expand(-1, -1,  -1, b.shape[2], -1)
    b1 = b.unsqueeze(2).expand(-1, -1, a.shape[2], -1,  -1)
    p = a1*b1
    y = p.sum(-1)  
    return y
    

def GEMM_fi(a, b, ber, bits):

    b = b.transpose(2,3)
    a1 = a.unsqueeze(3).expand(-1, -1,  -1, b.shape[2], -1)
    b1 = b.unsqueeze(2).expand(-1, -1, a.shape[2], -1,  -1)
    p = a1*b1
             
    p = operation_mulfi(p, ber, bits)
                
    c = torch.cumsum(p, dim=-1)
             
    c_fi = operation_fi(c, ber, bits)     
              
    y_sum = c_fi[:,:,:,:, -1,...]

    c_err = c_fi[:,:,:,:, 1:-1,...] - c[:,:,:, :,1:-1,...]

    c_error = c_err.sum(dim=-1)
            
    y = y_sum + c_error
         
    y = operation_fi(y, ber, bits) 
       
    return y  

  
 
 
 
 
     