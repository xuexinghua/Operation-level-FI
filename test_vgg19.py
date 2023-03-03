import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from layer.bitflip import bitFlip
import os

from models import *
from utils import progress_bar



device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 5

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=50, shuffle=False, num_workers=2)




print('==> Building model..')




#net = VGG('VGG19')

net = addmul_fi_VGG('VGG19')


net = net.to(device)

checkpoint = torch.load('./checkpoint/vgg19ckp.pth')



net.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()

def test(net,epoch):
    
    net.eval()
   
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
          
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

tloss=0
totalacc2 = 0
for epoch in range(0, epochs):    

    

    acc2,loss = test(net,epoch)
    totalacc2 += acc2
    tloss += loss


totalacc2 /= epochs
tloss /= epochs

print('-==================-vgg-==================-')

print('total win test acc: ',totalacc2)  
print(tloss)