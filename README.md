# Operation-level fault injection
OPFI(Operation Fault Injection) is an open-source Python tool for fault injection for multiplication and addition operations of DNNs. OPFI can support fault injection of various linear layers and nonlinear layers, and also supports mainstream convolution layer optimization algorithms such as winograd and FFT fault injection. Fault injection in OPFI is achieved through tasks performed on target operations, each task can correspond to a specific layer or application, which can be a baseline or a fault trigger.
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
