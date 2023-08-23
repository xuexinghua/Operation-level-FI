
import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
import math
from layer.winograd import transfor
import multiprocessing
from layer.quantization import Quant
from layer.fi import operation_fi, operation_mulfi


def matmul_a_fi(a, b, ber, bits):

    b = b.transpose(3, 4)
    a1 = a.unsqueeze(1).expand(-1, b.shape[3], -1)
    b1 = b.unsqueeze(3).expand(-1, -1, -1, a.shape[0], -1, -1)
    p = a1*b1
                
    p = operation_mulfi(p, ber, bits)
                
    c = torch.cumsum(p, dim=-1)
               
    c_fi = operation_fi(c, ber, bits)         
             
    y_sum = c_fi[:,:,:,:, :, -1,...]
    

    c_err = c_fi[:,:,:,:, :, 1:-1,...] - c[:,:,:, :, :,1:-1,...]

    c_error = c_err.sum(dim=-1)
           
    y = y_sum + c_error
                               
    y = operation_fi(y, ber, bits) 
       
    return y    

def matmul_b_fi(a, b, ber, bits):

    b = b.transpose(0, 1)
    a1 = a.unsqueeze(4).expand(-1, -1, -1, -1, b.shape[0], -1)
    b1 = b.unsqueeze(0).expand(a.shape[3], -1, -1)
    p = a1*b1
                
    p = operation_mulfi(p, ber, bits)
                
    c = torch.cumsum(p, dim=-1)
                
    c_fi = operation_fi(c, ber, bits)         
              
    y_sum = c_fi[:,:,:,:, :, -1,...]

    c_err = c_fi[:,:,:,:, :, 1:-1,...] - c[:,:,:, :, :,1:-1,...]

    c_error = c_err.sum(dim=-1)
            
    y = y_sum + c_error
                               
    y = operation_fi(y, ber, bits) 
       
    return y 

def tile(input,F,filterDim):
        
        chunk_dim = F + filterDim - 1
        s = F
        tiledShape = 0
        numChunks = 0

        tensor = input.unfold(2,chunk_dim,s)

        tensor = tensor.unfold(3, chunk_dim, s)

        tiledShape = tensor.shape
        numChunks = tensor.shape[2]
        return tiledShape,numChunks,tensor.contiguous().view(tensor.size(0), tensor.size(1), -1, tensor.size(4), tensor.size(5))




def untile(output,F,tiledShape,numChunks):
       
        output = output.reshape(output.shape[0], output.shape[1], tiledShape[2], tiledShape[3], F, F)        
        return output.transpose(4,3).contiguous().squeeze().view(output.shape[0], output.shape[1], F*numChunks, F*numChunks)


def pad_for_tiling(input,padding,F):
        
        number_tiling_positions = (input.shape[3] - 2*padding)/F
        if (number_tiling_positions).is_integer():
            Pad_tiling = torch.nn.ZeroPad2d(0)
        else:
            
            decimal_part = number_tiling_positions - int(number_tiling_positions)
            to_pad = round((1.0-decimal_part) * F)
            to_pad_even = round(to_pad/2)
            Pad_tiling = torch.nn.ZeroPad2d((to_pad_even, to_pad_even, to_pad_even, to_pad_even))
            print("Pad for tiling is {} for input {} and config F{}".format(Pad_tiling.padding, input.shape, F)) 
        return Pad_tiling
   
   
def win_conv2d(in_feature, kernel, padding, win_outtile, bits, stride, bias=None):		
    
    quant = Quant(bits)
    kernel = quant(kernel)     
    
    stride = stride 
    batch = in_feature.size(0)
    in_channel = in_feature.size(1)
    orig_h, orig_w = in_feature.size(2), in_feature.size(3)
    
    out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
    dh, dw = win_outtile, win_outtile
    win_input_size = win_outtile+2		
    padding = padding
    expected_output_width = ((orig_h + 2*padding - keh) // stride) + 1
    img = F.pad(input= in_feature, pad= (padding, padding, padding, padding), mode='constant', value= 0)    
    wifpad = (orig_w+2*padding-win_input_size)%dw
    hifpad = (orig_h+2*padding-win_input_size)%dh
    
    if wifpad == 0:
        rp = 0
        img = img
    else:  
        rp = dw- wifpad
        img = F.pad(input= img, pad= (0, rp, 0, 0), mode='constant', value= 0)         
    if hifpad == 0:
        dp = 0
        img = img
    else:
        dp = dh- hifpad
        img = F.pad(input= img, pad= (0, 0, 0, dp), mode='constant', value= 0)      		
    inh, inw = img.size(2), img.size(3)
    array_AT,array_BT,array_G,Tarray_AT,Tarray_BT,Tarray_G = transfor(win_outtile)      

    tiledShape,numChunks,input_ = tile(img,win_outtile,keh)
    
    input_ = input_.cuda()
    weight = kernel.unsqueeze(2)
    
    weight_winograd = torch.matmul(torch.matmul(array_G, weight), Tarray_G).cuda()  
    input_winograd = torch.matmul(torch.matmul(array_BT, input_),Tarray_BT).cuda()
    
    input_winograd = input_winograd.unsqueeze(1).expand(-1, weight_winograd.shape[0], -1, -1, -1, -1)    
    
    p = input_winograd * weight_winograd      
    point_wise = (p).sum(2)
     
    out = torch.matmul(array_AT, point_wise)

    output_ = torch.matmul(out, Tarray_AT)  

    output = untile(output_,win_outtile,tiledShape,numChunks)      
    if output.shape[3] is not expected_output_width:        
        output = output[:,:,:expected_output_width,:expected_output_width]
    out_h = output.shape[2]
    out_w = output.shape[3]        
    output = output.contiguous().view(output.shape[0], output.shape[1], output.shape[2]*output.shape[3]).transpose(1, 2)
    if bias is None:        
       output = output.transpose(1, 2)
    else:
       output = (output + bias).transpose(1, 2)    
    out = output.contiguous().view(batch, out_channel, out_h, out_w)     
    
    return out 
        
class winconv2d(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, win_outtile, bits, stride, padding, bias=None):
		super(winconv2d, self).__init__() 

		self.bits = bits
		self.stride = stride
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.win_outtile = win_outtile
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
  
		if bias:
						self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
						self.register_parameter('bias', None)

	def forward(self, x):    		
		return win_conv2d(x, self.weight, self.padding, self.win_outtile, self.bits, self.stride, self.bias)


def win_conv2d_fi(in_feature, kernel, padding, win_outtile, ber, bits, stride, bias=None):    

    quant = Quant(bits)
    kernel = quant(kernel)

    stride = stride
    batch = in_feature.size(0)
    in_channel = in_feature.size(1)
    orig_h, orig_w = in_feature.size(2), in_feature.size(3)
    
    out_channel, keh, kew = kernel.size(0), kernel.size(2), kernel.size(3)		
    dh, dw = win_outtile, win_outtile
    win_input_size = win_outtile+2		
    padding = padding
    expected_output_width = ((orig_h + 2*padding - keh) // stride) + 1
    img = F.pad(input= in_feature, pad= (padding, padding, padding, padding), mode='constant', value= 0)    
    wifpad = (orig_w+2*padding-win_input_size)%dw
    hifpad = (orig_h+2*padding-win_input_size)%dh
    if wifpad == 0:
        rp = 0
        img = img
    else:
        rp = dw- wifpad
        img = F.pad(input= img, pad= (0, rp, 0, 0), mode='constant', value= 0)    
     
    if hifpad == 0:
        dp = 0
        img = img
    else:
        dp = dh- hifpad
        img = F.pad(input= img, pad= (0, 0, 0, dp), mode='constant', value= 0)      		
    inh, inw = img.size(2), img.size(3)
    array_AT,array_BT,array_G,Tarray_AT,Tarray_BT,Tarray_G = transfor(win_outtile)      
         
    tiledShape,numChunks,input_ =  tile(img,win_outtile,keh)
    input_ = input_.cuda()
    weight = kernel.unsqueeze(2)      
    weight_winograd = torch.matmul(torch.matmul(array_G, weight), Tarray_G).cuda()          
    input_winograd = torch.matmul(torch.matmul(array_BT, input_),Tarray_BT).cuda()                   
    input_winograd = input_winograd.unsqueeze(1).expand(-1, weight_winograd.shape[0], -1, -1, -1, -1)               
    
    p = input_winograd * weight_winograd      
        
    p = operation_mulfi(p, ber, bits)          
    
    c = torch.cumsum(p, dim=2)        
    c_fi = operation_fi(c, ber, bits)       
    y_sum = c_fi[:,:,-1,...]    
    c_err = c_fi[:,:,1:-1,...] - c[:,:,1:-1,...]    
    c_error = c_err.sum(dim=2)             
    y = y_sum + c_error	      
    y = operation_fi(y, ber, bits)  
  
    #y = (p).sum(2)
        
    out = torch.matmul(array_AT, y)
    output_ = torch.matmul(out, Tarray_AT)          
    output = untile(output_,win_outtile,tiledShape,numChunks)      
    if output.shape[3] is not expected_output_width:        
        output = output[:,:,:expected_output_width,:expected_output_width]
    out_h = output.shape[2]
    out_w = output.shape[3]          
    output = output.contiguous().view(output.shape[0], output.shape[1], output.shape[2]*output.shape[3]).transpose(1, 2)
    if bias is None:        
       output = output.transpose(1, 2)
    else:
       output = (output + bias).transpose(1, 2)    
    out = output.contiguous().view(batch, out_channel, out_h, out_w)         
    
    #out = operation_fi(out, ber, bits)      

    return out 


class winconv2d_fi(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, win_outtile, ber, bits, stride, padding, bias=None):
		super(winconv2d_fi, self).__init__() 

		self.ber = ber
		self.bits = bits	
		self.stride = stride	
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.win_outtile = win_outtile
		self.padding = padding
		self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
 
		if bias:
						self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
		else:
						self.register_parameter('bias', None)

	def forward(self, x):    		
		return win_conv2d_fi(x, self.weight, self.padding, self.win_outtile, self.ber, self.bits, self.stride, self.bias) 








