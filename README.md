# Operation-level fault injection
OPFI(Operation Fault Injection) is an open-source Python tool for fault injection for multiplication and addition operations of DNNs. OPFI can support fault injection of various linear layers and nonlinear layers, and also supports mainstream convolution layer optimization algorithms such as winograd and FFT fault injection. Fault injection in OPFI is achieved through tasks performed on target operations, each task can correspond to a specific layer or application, which can be a baseline or a fault trigger.

## Summary of OPFI's capabilities
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
## Usage
### 1、Training
```ruby
python train.py
```
### 2、Evaluating
```ruby
python test_vgg19.py
```
In ```test_vgg19.py```, using ```net = VGG('VGG19')``` in line 33 to test the accuracy of the standard convolution without failure, using ```net = VGG_fi('VGG19')``` in line 34 to test the accuracy of the standard convolution with failure.
You can modify the ```ber = XXX``` in ```./layer/layers.py``` to set the bit error rate parameter.
