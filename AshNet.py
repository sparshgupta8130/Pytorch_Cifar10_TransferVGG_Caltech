
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# In[ ]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:


class MyNet(torch.nn.Module):
#constructor
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,48,(3,3))
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(48)
        
        self.conv2 = torch.nn.Conv2d(48,48,(3,3))
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.conv3 = torch.nn.Conv2d(48,96,(3,3))
        
        self.conv4 = torch.nn.Conv2d(96,96,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(96)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv5 = torch.nn.Conv2d(96,192,3)
        self.conv6 = torch.nn.Conv2d(192,192,3)
        
        
        #torch.nn.init.xavier_normal(self.conv4.weight)
        #self.bn4 = nn.BatchNorm2d(192)
        
        self.fc1 = torch.nn.Linear(4*4*192,500)
        
        #torch.nn.init.xavier_normal(self.fc1.weight)
        #self.bn5 = nn.BatchNorm1d(200)
        
        self.fc2 = torch.nn.Linear(500,500)
        
        #torch.nn.init.xavier_normal(self.fc2.weight)
        #self.bn6 = nn.BatchNorm1d(200)
        self.fc3 = torch.nn.Linear(500,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1,4*4*192)
        x = (F.relu(self.fc1(x)))
        x = (F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# In[ ]:


net = MyNet()
net.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9,0.999))


# In[ ]:


for epoch in range(50):
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


# In[ ]:


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


# In[ ]:


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

