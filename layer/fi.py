
from layer.bitflip import int2bin,bin2int
from bitstring import BitArray
import torch
import numpy as np
import math 

error_model = 'random_bit_flip'
#error_model = 'specified_bit_flip'

flip_bit_position = 7

def quan_FI(p,ab,bit):
    
    bit = int(bit)
    fiinput = torch.reshape(p,(-1,)).clone()  
    num = fiinput.shape[-1]    
    bitnum = num*bit
    w = ab*bitnum      
    finum = round(w)
    if finum==0:
       fiinput = fiinput
    else:             
        index = np.random.randint(0,bitnum,size=finum)
        allbitindex = torch.from_numpy(index).cuda()    
        valueindex = allbitindex//bit                       
        value = fiinput.index_select(0,valueindex)    
        
        if error_model == 'random_bit_flip':
             valuebitindex = allbitindex - valueindex*bit
        elif error_model == 'specified_bit_flip':
             val_len = value.shape[-1]
             valuebitindex = torch.zeros(val_len).cuda()
             index = flip_bit_position 
             valuebitindex = valuebitindex + index  
             
        value_bin1 = int2bin(value, bit).char()
        bit_mask = (value_bin1.clone().zero_() + 1) * (2**valuebitindex).char()          
        bin_w = value_bin1 ^ bit_mask
        fivalue = bin2int(bin_w, bit).float()                       
        fiinput = fiinput.index_put_((valueindex,), fivalue)
    return fiinput

def quan_mulFI(p,ab,bit):

    bit = int(bit)
    fiinput = torch.reshape(p,(-1,)).clone()  
    num = fiinput.shape[-1]    
    bitnum = num*bit
    w = 2*ab*bitnum  
    finum = round(w)
    if finum==0:
       fiinput = fiinput
    else:             
        index = np.random.randint(0,bitnum,size=finum)
        allbitindex = torch.from_numpy(index).cuda()    
        valueindex = allbitindex//bit                       
        value = fiinput.index_select(0,valueindex)    
        
        if error_model == 'random_bit_flip':
             valuebitindex = allbitindex - valueindex*bit
        elif error_model == 'specified_bit_flip':
             val_len = value.shape[-1]
             valuebitindex = torch.zeros(val_len).cuda()
             index = flip_bit_position 
             valuebitindex = valuebitindex + index            
        
        value_bin1 = int2bin(value, bit).char()
        bit_mask = (value_bin1.clone().zero_() + 1) * (2**valuebitindex).char()          
        bin_w = value_bin1 ^ bit_mask
        fivalue = bin2int(bin_w, bit).float()                       
        fiinput = fiinput.index_put_((valueindex,), fivalue)
    return fiinput

    
def operation_fi(x, ber, bits):       
    
    y = x.shape
    z = len(y)    
    if z == 2:
        b = x.shape[0]        
        h = x.shape[1]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h))        
    elif z == 3:
        b = x.shape[0]        
        h = x.shape[1]
        w = x.shape[2]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w))        
    elif z == 4:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f))    
    elif z == 5:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m))            
    elif z == 6:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m,n))            

    elif z == 7:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        o = x.shape[6]
        fiinput = quan_FI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m,n,o))                  
    return out    
 
def operation_mulfi(x, ber, bits):       
    
    y = x.shape
    z = len(y)    
    if z == 2:
        b = x.shape[0]        
        h = x.shape[1]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h))        
    elif z == 3:
        b = x.shape[0]        
        h = x.shape[1]
        w = x.shape[2]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w))        
    elif z == 4:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f))    
    elif z == 5:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m))            
    elif z == 6:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m,n))            
    elif z == 7:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        o = x.shape[6]
        fiinput = quan_mulFI(x, ber, bits)
        out = torch.reshape(fiinput,(b,h,w,f,m,n,o))                  
    return out      
