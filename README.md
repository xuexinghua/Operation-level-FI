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
In test_vgg19.py, using net = VGG('VGG19') in line 33 to test the accuracy of the standard convolution without failure, using net = VGG_fi('VGG19') in line 34 to test standard convolutional when there is a failure
