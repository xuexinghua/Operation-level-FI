import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from layer.bitflip import bitFlip
import os
import argparse
from models_activatefi import *
from utils import progress_bar
from data_loader import test_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parsers
parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--net', default='VGG')
parser.add_argument('--n_epochs', type=int, default='1')
parser.add_argument('--ber', nargs='+', type=float, default="[1e-10]")
parser.add_argument('--n_bit', default="16")
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--batch', type=int, default='100')

args = parser.parse_args()
epochs= args.n_epochs
bits = args.n_bit
BER = args.ber
batch = args.batch


print('==> Preparing data..')
if args.dataset=="cifar10":

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        
        n = 1500
        a = len(testset)- n
        testset, other = torch.utils.data.random_split(testset,[n,a])  
                       

            
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif args.dataset=="cifar100":

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        
        n = 1500
        a = len(testset)- n
        testset, other = torch.utils.data.random_split(testset,[n,a])      
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch, shuffle=False, num_workers=2)


elif args.dataset=="imagenet":

        data = '/home/xuexinghua/github/operation_fi/data/imagenet/ILSVRC2012/'
        testloader = test_loader(data, batch_size=batch, workers=2)


criterion = nn.CrossEntropyLoss()


def test(epoch, ber, bit):
   
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
                        
            
            if args.net=="vgg19_BatchNorm2d_fi":
                net = BatchNorm2d_fi_VGG('VGG19', ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/vgg19ckp.pth')
                net.load_state_dict(checkpoint)
            
            elif args.net=="vgg19_ReLU_fi":
                net = ReLU_fi_VGG('VGG19', ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/vgg19ckp.pth')
                net.load_state_dict(checkpoint)

            elif args.net=="vgg19_MaxPool2d_fi":
                net = MaxPool2d_fi_VGG('VGG19', ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/vgg19ckp.pth')
                net.load_state_dict(checkpoint)
                
            elif args.net=="vgg19_AvgPool2d_fi":
                net = AvgPool2d_fi_VGG('VGG19', ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/vgg19ckp.pth')
                net.load_state_dict(checkpoint)
                

                                
            elif args.net=="resnet_BatchNorm2d_fi":
                net = BatchNorm2d_fi_resnet50(ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/ResNet50ckp.pth')
                net.load_state_dict(checkpoint)

            elif args.net=="resnet_ReLU_fi":
                net = ReLU_fi_resnet50(ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/ResNet50ckp.pth')
                net.load_state_dict(checkpoint)

            elif args.net=="resnet_AvgPool2d_fi":
                net = AvgPool2d_fi_resnet50(ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/ResNet50ckp.pth')
                net.load_state_dict(checkpoint)

            elif args.net=="resnet_MaxPool2d_fi":
                net = MaxPool2d_fi_resnet50(ber, bit).to(device)
                checkpoint = torch.load('./checkpoint/ResNet50ckp.pth')
                net.load_state_dict(checkpoint)



            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(predicted)
            loss = test_loss/(batch_idx+1)
    
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         
    acc = 100.*correct/total

    return acc,loss



for ber in BER:
                      
         
         print('------BER----:', ber)
         
         tacc = 0        
         
         for epoch in range(epochs):
 
             acc, val_loss= test(epoch, ber, bits)
             
             print('epoch: %d,  accurancy: %d'%(epoch, acc))
                          
             tacc += acc 
                                               
         tacc /= epochs                
         
         print( 'BER: %d, average accurancy: %d'%(ber, tacc))
         






