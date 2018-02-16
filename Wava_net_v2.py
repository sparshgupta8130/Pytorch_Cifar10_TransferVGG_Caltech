
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
get_ipython().magic('matplotlib inline')
np.random.seed(0)


# In[2]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


class MyNet(torch.nn.Module):
#constructor
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,3,padding=1)
        nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32,64,3)
        nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,64,3)
        self.bn3  = nn.BatchNorm2d(64)
        nn.init.xavier_normal(self.conv3.weight)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        nn.init.xavier_normal(self.conv4.weight)
        
        self.conv5 = nn.Conv2d(128,128,3)
        self.bn5 = nn.BatchNorm2d(128)
        nn.init.xavier_normal(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(128,256,3,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        nn.init.xavier_normal(self.conv6.weight)
        
        self.conv7 = nn.Conv2d(256,256,3)
        self.bn7 = nn.BatchNorm2d(256)
        nn.init.xavier_normal(self.conv7.weight)
        
        self.fc1  = torch.nn.Linear(1024,1024)
        nn.init.xavier_normal(self.fc1.weight)
        self.bn8 = nn.BatchNorm1d(1024)
        
        self.fc2  = torch.nn.Linear(1024,1024)
        nn.init.xavier_normal(self.fc1.weight)
        self.bn9 = nn.BatchNorm1d(1024)
        
        self.fc3  = torch.nn.Linear(1024,10)
        
        
    def forward(self,x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(self.bn7(F.relu(self.conv7(x))))
        x = x.view(-1,2*2*256)
        
        x = self.bn8(F.relu(self.fc1(x)))
        x = self.bn9(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# In[4]:


net = MyNet()
net.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9,0.999))


# In[5]:


for epoch in range(200):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if i%2000 == 1999:
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
print('Finished Training.')
correct = 0
total = 0
for data in testloader:
    images, labels = data
    #images,labels = Variable(inputs.cuda()),Variable(labels.cuda())
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.cuda().size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[6]:


correct = 0
total = 0
for data in testloader:
    images, labels = data
    #images,labels = Variable(inputs.cuda()),Variable(labels.cuda())
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.cuda().size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[7]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    #images,labels = Variable(inputs.cuda()),Variable(labels.cuda())
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

