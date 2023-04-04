
import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
import math
import multiprocessing
from layer.bitflip import bit2float, float2bit
from bitstring import BitArray



ber = 1e-6



def FI(p,ab):

    fiinput = torch.reshape(p,(-1,))  
    num = fiinput.shape[-1] 
    bitnum = num*32
    w = bitnum*ab       
    finum = math.ceil(w)   

    if finum==0:
       fiinput = fiinput
    else:
            
       index = np.random.randint(0,bitnum,size=finum)
       allbitindex = torch.from_numpy(index).cuda()    
       valueindex = allbitindex//32
       value = fiinput.index_select(0,valueindex)
       valuebitindex = allbitindex - valueindex*32
       encodernum = valuebitindex.shape[-1]
       valencoderidx = torch.arange(0,encodernum, 1).cuda()
       valbitencoder = valuebitindex + valencoderidx*32    
       value= torch.reshape(value,(1,-1))
       value_bin1 = float2bit(value, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32)
       value_bin2 = torch.reshape(value_bin1,(-1,))    
       bit_value1 = value_bin2.index_select(0,valbitencoder)
       error_bit = torch.abs(bit_value1-1)        
       bin_value = value_bin2.index_put_((valbitencoder,), error_bit) 
       nm = int(bin_value.shape[-1]/32)  
       bin_value1 = torch.reshape(bin_value,(nm,32))         
       fivalue = bit2float(bin_value1, num_e_bits=8, num_m_bits=23, bias=127.) 
       fiinput = fiinput.index_put_((valueindex,), fivalue)
    return fiinput
    


def diret_conv2d(in_feature, kernel,  padding, bias=None):
           
        fiinputs = []
        batch = in_feature.size(0)
        in_channel = in_feature.size(1)
        orig_h, orig_w = in_feature.size(2), in_feature.size(3)
        out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
        padding = padding
        stride=1
        out_rows = ((orig_h + 2*padding - keh) // stride) + 1
        out_cols = ((orig_w + 2*padding - kew) // stride) + 1
        inp_unf = torch.nn.functional.unfold(in_feature, (keh, kew), padding=padding)
        w = kernel.contiguous().view(kernel.size(0), -1).t()
        x = inp_unf.transpose(1, 2)    	
	x = x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
        w = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)	
	p = x * w
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

	def __init__(self, in_channels, out_channels, kernel_size, padding,bias=None):
		super(conv2d, self).__init__()		
		
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels,kernel_size, kernel_size))

		if bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
			self.register_parameter('bias', None)
	def forward(self, x):
		return diret_conv2d(x, self.weight,  self.padding, self.bias)



def diret_conv2d_fi(in_feature, kernel, padding, bias=None):
         
    batch = in_feature.size(0)
    in_channel = in_feature.size(1)
    orig_h, orig_w = in_feature.size(2), in_feature.size(3)
    out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
    padding = padding
    stride=1
    out_rows = ((orig_h + 2*padding - keh) // stride) + 1
    out_cols = ((orig_w + 2*padding - kew) // stride) + 1
       
    inp_unf = torch.nn.functional.unfold(in_feature, (keh, kew), padding=padding)        
    
    w = kernel.contiguous().view(kernel.size(0), -1).t()
    x = inp_unf.transpose(1, 2)                
         
    x1= x.unsqueeze(1).expand(-1, w.shape[1], -1, -1)
    w1 = w.transpose(0,1).unsqueeze(1).expand(-1, x.shape[1], -1)

    p=x1*w1      
                         
    #-------------------    
    b = p.shape[0]
    a = p.shape[1]
    h = p.shape[2]
    w = p.shape[3]       
    fiinput = FI(p, ber)     
    p = torch.reshape(fiinput, (b,a,h,w))
    #-------------------      
                  
    y= (p).sum(3)        
    
    #-------------------  
    b1 = y.shape[0]
    a1 = y.shape[1]
    h1 = y.shape[2]
    fiinput = FI(y, ber)                   
    y = torch.reshape(fiinput, (b1,a1,h1))     
    #-------------------             
      
    y = y.transpose(1,2)            
    if bias is None:        
         out_unf = y.transpose(1, 2)
    else:
         out_unf = (y + bias).transpose(1, 2)        
    out = out_unf.contiguous().view(batch, out_channel, out_rows,out_cols)
   
    return out

class conv2d_fi(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, padding,bias=None):
		super(conv2d_fi, self).__init__()		
		#Initialize misc. variables
   

		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels,kernel_size, kernel_size))
 
		if bias:
						self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
						self.register_parameter('bias', None)
	def forward(self, x):
		return diret_conv2d_fi(x, self.weight, self.padding, self.bias)        




