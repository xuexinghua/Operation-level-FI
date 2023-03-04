# Operation-level fault injection
Failure analysis on the data of the multiplication-addition calculation process, which can distinguish different operator implementations (e.g normal convolution, winograd convolution and FFT convolution).
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
