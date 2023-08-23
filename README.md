# Operation-level fault injection
OPFI(Operation Fault Injection) is an open-source Python tool for fault injection for multiplication and addition operations of DNNs. OPFI can support fault injection of various linear layers and nonlinear layers, and also supports mainstream convolution layer optimization algorithms such as winograd and FFT fault injection. Fault injection in OPFI is achieved through tasks performed on target operations, each task can correspond to a specific layer or application, which can be a baseline or a fault trigger.

## OPFI will support the following:
As of now, OPFI can inject errors in the following layer operations.
* Linear layer
  * Convolutional layer
    * Direct convolution
    * Winograd convolution
    * FFT convolution
  * FC layer
  * GEMM
* Nonlinear layer
  * ReLU
  * GELU
  * Softmax
  * AvgPool2d
  * MaxPool2d
  * BatchNorm
  * LayerNorm
  
Currently, we have two bit-flip models implemented:
* random_bit_flip: Flip randomly selected bits from 0 to 1 or 1 to 0
* specified_bit_flip: Flip the specified bit from 0 to 1 or 1 to 0

More custom layers can be added by modifying ```./layer/conv_layers.py``` or ```./layer/linear_layers.py``` or ```./layer/activate_layer.py```, New bit-flip models can be added by modifying ```./layer/fi.py```
## Requirements
* Python (3.7)
* pytorch（1.3.1）
* bitstring（3.1.9）
* torchvision（0.4.2）
## Getting Started
Clone the repository
```ruby
git clone https://github.com/xuexinghua/Operation-level-FI.git
```

Then, download the trained model file and put it into the ```checkpoint/``` folder. Put the downloaded dataset in  ```data/``` folder. 

### Some examples of running fault injection are as follows:
1、Single layer fault injection

```ruby
CUDA_VISIBLE_DEVICES=XXX python test_diff_layer.py  [ --layertype LAYERTYPE ] [ --ber BITERRORRATE ]
```

Its optional arguments are the following:
* --layertype: Specify the layer that needs to perform fault injection (direct_conv, win_conv, fft_conv, fc, gemm, ReLU, GELU, softmax, avgpool2d, maxpool2d, etc)
* --ber: Bit Error Rate (e.g., 1E-6)

2、Model Fault Injection

Fault Injection for Linear Layers in Model
```ruby
CUDA_VISIBLE_DEVICES=XXX python test_model_linearfi.py [ --net MODEL ] [ --dataset DATASET ] [ --ber BITERRORRATE ]
```

Fault Injection for Nonlinear Layers in Model
```ruby
CUDA_VISIBLE_DEVICES=XXX python test_model_nonlinearfi.py  [ --net MODEL ] [ --dataset DATASET ] [ --ber BITERRORRATE ]
```

Its optional arguments are the following:
* --net: When running the script ```test_model_linearfi.py```, you can choose vgg19_fi, winvgg19_fi, fftvgg19_fi, resnet_fi, etc. When running the script ```test_model_nonlinearfi.py```, you can choose vgg19_ReLU_fi, vgg19_BatchNorm2d_fi, vgg19_MaxPool2d_fi, resnet_AvgPool2d_fi, etc.
* --dataset: In the case given in the ```models/``` or ```models_activate/``` folder, the vgg model use the cifar100 dataset, and the resnet model use the imagenet dataset.
* --ber: Bit Error Rate. （e.g., 1E-6）

The bit-flip model can be selected by modifying ```error_model``` in line 8 or line9 in ```./layer/fi.py```
