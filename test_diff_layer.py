
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F

from layer.conv_layers import diret_conv2d_fi, diret_conv2d
from layer.winograd_layers import win_conv2d_fi, win_conv2d
from layer.fft_layer import fft_conv, fi_fft_conv
from layer.activate_layer import relu, relu_fi, maxpool2d, maxpool2d_fi, avgpool2d, avgpool2d_fi, gelu, gelu_fi, softmax, softmax_fi
from layer.linear_layers import GEMM_fi, GEMM, quan_linear_fi, quan_linear
import argparse


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--layertype', default='direct_conv')
parser.add_argument('--n_bit', default="16")
parser.add_argument('--ber', nargs='+', type=float, default="[1e-10]")

args = parser.parse_args()
bits = args.n_bit
BER = args.ber
tiles = 2


def rmse(y_noerror, y):
       
        RMSE = torch.sqrt(torch.mean((y - y_noerror).pow(2)))                
        errornum = 1- torch.isclose(y_noerror, y, rtol=2e-3, atol=1e-3).long()                        
        errornum = len(torch.nonzero(errornum))
  
        return RMSE, errornum
        
def test(ber, bit):

    if args.layertype=="direct_conv":

        # direct convolution FI
        diret_conv2d_noerror = diret_conv2d(x, weight, padding, bits, stride, bias)
        diret_conv2d_error = diret_conv2d_fi(x, weight, padding, ber, bits, stride, bias)
        RMSE, errornum = rmse(diret_conv2d_noerror, diret_conv2d_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="win_conv":
    
        # winograd convolution FI
        win_conv2d_noerror = win_conv2d(x, weight, padding, tiles, bits, stride, bias)
        win_conv2d_error = win_conv2d_fi(x, weight, padding, tiles, ber, bits, stride, bias)
        RMSE, errornum = rmse(win_conv2d_noerror, win_conv2d_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="fft_conv":
    
        # FFT convolution FI
        fft_conv2d_noerror = fft_conv(bits, x, weight, padding=padding, stride=stride, bias=bias)
        fft_conv2d_error = fi_fft_conv(ber, bits, x, weight, padding=padding, stride=stride, bias=bias)
        RMSE, errornum = rmse(fft_conv2d_noerror, fft_conv2d_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))


    elif args.layertype=="ReLU":
    
        # ReLU fi
        ReLU_noerror = relu(x, bits)         
        print(ReLU_noerror)
        ReLU_error = relu_fi(x, ber, bits)
        print(ReLU_error)
        RMSE, errornum = rmse(ReLU_noerror, ReLU_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="GELU":
    
        # GELU fi
        GELU_noerror = gelu(x, bits)
        GELU_error = gelu_fi(x, ber, bits)
        RMSE, errornum = rmse(GELU_noerror, GELU_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="softmax":
    
        # softmax fi
        softmax_noerror = softmax(x, 3, bits)
        softmax_error = softmax_fi(x, 3, ber, bits)
        RMSE, errornum = rmse(softmax_noerror, softmax_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="avgpool2d":
    
        # avgpool2d fi
        avgpool2d_noerror = avgpool2d(x, bits, 2, 2)
        avgpool2d_error = avgpool2d_fi(x, ber, bits, 2, 2)
        RMSE, errornum = rmse(avgpool2d_noerror, avgpool2d_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="maxpool2d":
    
        # maxpool2d fi
        maxpool2d_noerror = maxpool2d(x, bits, 2, stride, padding)
        maxpool2d_error = maxpool2d_fi(x, ber, bits, 2, stride, padding)
        RMSE, errornum = rmse(maxpool2d_noerror, maxpool2d_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

    elif args.layertype=="gemm":
    
        # GEMM fi
        gemm_noerror = GEMM(a, b)
        print(gemm_noerror)
        gemm_error = GEMM_fi(a, b, ber, bits)
        print(gemm_error)
        RMSE, errornum = rmse(gemm_noerror, gemm_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))



    elif args.layertype=="fc":
    
        # GEMM fi
        fc_noerror = quan_linear(x, weight, bits, bias)
        
        fc_error = quan_linear_fi(x, weight, ber, bits, bias)
        
        RMSE, errornum = rmse(fc_noerror, fc_error)
        print( 'ErrorNum: %d, RMSE: %d'%(errornum, RMSE))

         
    return RMSE, errornum


if args.layertype=="gemm":
    a = torch.randn(2, 2, 32, 64).cuda() 
    b = torch.randn(2, 2, 64, 32).cuda() 


elif args.layertype=="fc":
    x = torch.randn(2, 32, 64).cuda() 
    weight = torch.randn(128, 64).cuda() 
    bias = torch.randn(128).cuda()


else:
    x = torch.randn(1, 64, 32, 32).cuda() 
    weight = torch.randn(64, 64, 3, 3).cuda() 
    bias = torch.randn(64).cuda()

    padding = 1 
    stride = 1





for ber in BER:

    print( 'BER: %d'%(ber))
    RMSE, errornum = test(ber, bits)                     







