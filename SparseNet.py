
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
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
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,(5,5))
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(6,16,(5,5))
        self.fc1 = torch.nn.Linear(16*5*5,120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,10)
        
    def forward(self,x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[4]:


class Sparsh_10_3xav(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xav,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(8,8,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(8,16,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(16,16,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(16,32,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(32,32,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(64,256)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(256,256)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(256,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv4(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv5(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv6(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv7(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,64)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[5]:


class Sparsh_10_3xav_2(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xav_2,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(8,16,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(64,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(128,256,3)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(256,512)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,512)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(512,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv4(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv5(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv6(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv7(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,256)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[6]:


class Sparsh_10_3xavbn(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xavbn,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(8,16,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(64,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(128,256,3)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(256,512)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,512)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(512,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bnfc1 = torch.nn.BatchNorm1d(512)
        self.bnfc2 = torch.nn.BatchNorm1d(512)
        
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.bn4(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.bn5(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.bn6(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv7(x))
        x = self.bn7(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,256)
        x = self.fc1(x)
        x = self.bnfc1(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.bnfc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[7]:


class Sparsh_10_3xavbnleak(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xavbnleak,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(8,16,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(64,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(128,256,3)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(256,512)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,512)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(512,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bnfc1 = torch.nn.BatchNorm1d(512)
        self.bnfc2 = torch.nn.BatchNorm1d(512)
        
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = self.bn4(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.bn5(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv6(x))
        x = self.bn6(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv7(x))
        x = self.bn7(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,256)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = self.bnfc1(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.bnfc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[8]:


class Sparsh_10_3xav_3(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xav_3,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(64,128,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(256,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.conv8 = torch.nn.Conv2d(256,512,3)
        torch.nn.init.xavier_normal(self.conv8.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(512,1024)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(1024,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(1024,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv4(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv5(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv6(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv7(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv8(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,512)
        x = torch.nn.functional.relu(self.fc1(x))
        #print(x.shape)
        x = torch.nn.functional.relu(self.fc2(x))
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[9]:


class Sparsh_10_3xavbn_3(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xavbn_3,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(64,128,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(256,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.conv8 = torch.nn.Conv2d(256,512,3)
        torch.nn.init.xavier_normal(self.conv8.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(512,1024)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(1024,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(1024,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.bnfc1 = torch.nn.BatchNorm1d(1024)
        self.bnfc2 = torch.nn.BatchNorm1d(1024)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.bn4(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.bn5(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.bn6(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv7(x))
        x = self.bn7(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.conv8(x))
        x = self.bn8(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,512)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.bnfc1(x)
        #print(x.shape)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.bnfc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[10]:


class Sparsh_10_3xavbnleak_3(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xavbnleak_3,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(32,64,3)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(64,128,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = torch.nn.Conv2d(128,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.conv7 = torch.nn.Conv2d(256,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.conv8 = torch.nn.Conv2d(256,512,3)
        torch.nn.init.xavier_normal(self.conv8.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(512,1024)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(1024,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(1024,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.bnfc1 = torch.nn.BatchNorm1d(1024)
        self.bnfc2 = torch.nn.BatchNorm1d(1024)
        
    def forward(self,x):
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = self.bn4(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.bn5(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv6(x))
        x = self.bn6(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv7(x))
        x = self.bn7(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.conv8(x))
        x = self.bn8(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,512)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = self.bnfc1(x)
        #print(x.shape)
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.bnfc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x


# In[11]:


class Sparsh_10_3xav_4(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xav_4,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,3,stride=2)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(16,32,3,padding=1)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(32,64,3,stride=2)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(64,128,3,padding=1)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(128,256,3,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(256,512)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(1024,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1,256)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[12]:


class Sparsh_10_3xav_5(torch.nn.Module):
    def __init__(self):
        super(Sparsh_10_3xav_5,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,16,7)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(32,64,5)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(64,128,3)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = torch.nn.Conv2d(128,256,3,stride=2,padding=1)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(256,512)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = torch.nn.Linear(512,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = torch.nn.Linear(1024,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1,256)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[13]:


def train_model(archi,lr=0.001,epochs=100):
    net = archi()
    net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.999))
    eps = []
    tr_acc = []
    te_acc = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
            #inputs,labels = Variable(inputs),Variable(labels)
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
        correct = 0
        total = 0
        for data in trainloader:
            images, labels = data
            #images, labels = Variable(images.cuda()),Variable(labels.cuda())
            outputs = net(Variable(images.cuda()))
            #outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cuda().size(0)
            #total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
            #correct += (predicted == labels).sum()
        #print('Accuracy after epoch %d : '%epoch,100*correct/total)
        tr_acc.append(100*correct/total)
        #eps.append(epoch)
        
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            #images, labels = Variable(images.cuda()),Variable(labels.cuda())
            outputs = net(Variable(images.cuda()))
            #outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cuda().size(0)
            #total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
            #correct += (predicted == labels).sum()
        #print('Accuracy after epoch %d : '%epoch,100*correct/total)
        te_acc.append(100*correct/total)
        eps.append(epoch)
        
        if epoch%10 == 0:
            print('Accuracy after epoch %d : '%epoch,te_acc[-1])
        
    print('Finished Training.')
    
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        #images, labels = Variable(images.cuda()),Variable(labels.cuda())
        outputs = net(Variable(images.cuda()))
        #outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        #total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        #correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        #images, labels = Variable(images.cuda()),Variable(labels.cuda())
        outputs = net(Variable(images.cuda()))
        #outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        #c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
    return eps,tr_acc,te_acc


# In[14]:


train_model(Sparsh_10_3xav_3,lr=0.0001,epochs=10)


# In[ ]:


train_model(Sparsh_10_3xavbnleak_3,lr=0.0001,epochs=50)


# In[ ]:


train_model(Sparsh_10_3xav_4,lr=0.0001,epochs=10)


# In[ ]:


train_model(Sparsh_10_3xav_5,lr=0.0001,epochs=10)


# In[14]:


eps,tr_acc,te_acc = train_model(Sparsh_10_3xavbnleak_3,lr=0.005,epochs=150)


# In[14]:


eps,tr_acc,te_acc = train_model(Sparsh_10_3xavbnleak_3,lr=0.005,epochs=60)

plt.plot(eps,tr_acc)
plt.plot(eps,te_acc)
plt.grid(True)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train Acc','Test Acc'],loc='lower right')
plt.show()

