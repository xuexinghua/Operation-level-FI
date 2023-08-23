from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn
from layer.quantization import Quant
from layer.bitflip import int2bin,bin2int
from bitstring import BitArray
import torch
import numpy as np
from layer.fi import operation_fi

def quan_FI(p, bits, finum):
    
    bits = int(bits)
    fiinput = torch.reshape(p,(-1,))  
    num = fiinput.shape[-1]    
    bitnum = num*bits
    if finum==0:
       fiinput = fiinput
    else:             
        index = np.random.randint(0,bitnum,size=finum)
        allbitindex = torch.from_numpy(index).cuda()    
        valueindex = allbitindex//bits      
        value = fiinput.index_select(0,valueindex)    
        valuebitindex = allbitindex - valueindex*bits        
        value_bin1 = int2bin(value, bits).short()   
        bit_mask = (value_bin1.clone().zero_() + 1) * (2**valuebitindex).short()  
        bin_w = value_bin1 ^ bit_mask
        fivalue = bin2int(bin_w, bits).float()
        #print(fivalue)
        fiinput = fiinput.index_put_((valueindex,), fivalue)
    return fiinput
    
    
def opt_fi(x, bits, num):       
    
    y = x.shape
    z = len(y)    
    if z == 2:
        b = x.shape[0]        
        h = x.shape[1]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h))        
    elif z == 3:
        b = x.shape[0]        
        h = x.shape[1]
        w = x.shape[2]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h,w))        
    elif z == 4:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h,w,f))    
    elif z == 5:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h,w,f,m))            
    elif z == 6:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h,w,f,m,n))            

    elif z == 7:
        b = x.shape[0]        
        h = x.shape[1]        
        w = x.shape[2]
        f = x.shape[3]
        m = x.shape[4]
        n = x.shape[5]
        o = x.shape[6]
        fiinput = quan_FI(x, bits, num)
        out = torch.reshape(fiinput,(b,h,w,f,m,n,o))                  
    return out  

#https://github.com/jkuli-net/ConvFFT/tree/main
#https://github.com/fkodom/fft-conv-pytorch/tree/master
def complex_matmul(a, b, groups = 1):

    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))
    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag
    return c.view(c.size(0), -1, *c.shape[3:])
    
def to_ntuple(val, n):

    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)

def fft_conv(bits, signal, kernel, bias = None, padding = 0, padding_mode = "constant", stride = 1, dilation = 1, groups = 1):

    
    quant = Quant(bits)
    kernel = quant(kernel)         
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bit, padding = 0, padding_mode = "constant", stride = 1, dilation = 1, groups = 1,  bias = True, ndim = 1):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.bit = bit
        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):
        return fft_conv(
            self.bit,
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


FFTConv1d = partial(_FFTConv, ndim=1)
FFTConv2d = partial(_FFTConv, ndim=2)
FFTConv3d = partial(_FFTConv, ndim=3)



def matmul_fi(a, b, ber, bits, num, mulnum):

    b = b.transpose(3, 4)
    a1 = a.unsqueeze(5).expand(-1, -1, -1, -1, -1, b.shape[3], -1)
    b1 = b.unsqueeze(3).expand(-1, -1, -1, a.shape[4], -1, -1)
    p = a1*b1        
    p = opt_fi(p, bits, mulnum)             
    c = torch.cumsum(p, dim=-1)             
    c_fi = opt_fi(c, bits, num)                    
    y_sum = c_fi[:,:,:,:,:,:, -1,...]
    c_err = c_fi[:,:,:,:,:,:, 1:-1,...] - c[:,:,:,:,:,:, 1:-1,...]
    c_error = c_err.sum(dim=-1)      
    y = y_sum + c_error                         
    y = operation_fi(y, ber, bits) 
       
    return y    


def fi_num(a, b, n_op, ber, bits):
    
    bits = int(bits)
    fiinput = torch.reshape(a,(-1,))  
    num = round(fiinput.shape[-1] * b.shape[-1] * n_op * bits * ber)   
    num = round(num /n_op)
    return num

def fi_mulnum(a, b, n_op, ber, bits):
    
    bits = int(bits)
    fiinput = torch.reshape(a,(-1,))  
    num = round(fiinput.shape[-1] * b.shape[-1] * n_op * bits * ber * 2)   
    num = round(num /n_op)
    return num
            

def complex_matmul_fi(ber, bits, a, b, groups = 1):

    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])
    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))
    num = fi_num(a.real, b.real, 4, ber, bits)
    mulnum = fi_mulnum(a.real, b.real, 4, ber, bits)
    # complex value matrix multiplication
    real_a = matmul_fi(a.real, b.real, ber, bits, num, mulnum)    
    real_b = matmul_fi(a.imag, b.imag, ber, bits, num, mulnum)    
    real = real_a - real_b    
    imag_a = matmul_fi(a.imag, b.real, ber, bits, num, mulnum)    
    imag_b = matmul_fi(a.real, b.imag, ber, bits, num, mulnum)    
    imag = imag_a + imag_b
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)   
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def fi_fft_conv(ber, bits, signal, kernel, bias = None, padding = 0, padding_mode = "constant", stride = 1, dilation = 1, groups = 1):
    
    quant = Quant(bits)
    kernel = quant(kernel)     
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))
    kernel_fr.imag *= -1  
    output_fr = complex_matmul_fi(ber, bits, signal_fr, kernel_fr, groups=groups)
    #print(output_fr)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    #print(output)
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class fi_FFTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ber, bit, stride = 1, padding = 0, padding_mode = "constant", dilation = 1, groups = 1,  bias = True, ndim = 1):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.bit = bit
        self.ber = ber
        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )
        kernel_size = to_ntuple(kernel_size, ndim)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):
        return fi_fft_conv(self.ber, self.bit, signal, self.weight, bias=self.bias, padding=self.padding, padding_mode=self.padding_mode, stride=self.stride, dilation=self.dilation,  groups=self.groups,)

FFTConv1d_fi = partial(fi_FFTConv, ndim=1)
FFTConv2d_fi = partial(fi_FFTConv, ndim=2)
FFTConv3d_fi = partial(fi_FFTConv, ndim=3)
