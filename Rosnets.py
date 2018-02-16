
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np


# In[4]:


def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))


# In[5]:


dataiter = iter(trainloader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[6]:


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[10]:


def train_test(cnn,eta,ep):
    loss = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(),lr=eta)

    for epoch in range(10):

        l_tr= 0.0
        for i,data in enumerate(trainloader,0):

            inputs,labels = data
            print(labels)
            inputs,labels = Variable(inputs),Variable(labels)
            optimiser.zero_grad()

            preds = cnn(inputs)
            print(preds)
            loss_tr = loss(preds,labels)
            loss_tr.backward()
            optimiser.step()

            l_tr += loss_tr.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, l_tr / 2000))
                l_tr = 0.0
    print ('finished training')
    
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = cnn(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = cnn(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# In[11]:


class rosnet_11_3(nn.Module):
    def __init__(self):
        super(rosnet_11_3,self).__init__()
        self.conv1 = nn.Conv2d(3,6,3,2,1)
        self.conv2 = nn.Conv2d(6,12,3,1,1)
        self.conv3 = nn.Conv2d(12,24,3,1,1)
        self.conv4 = nn.Conv2d(24,48,3,1,1)
        self.conv5 = nn.Conv2d(48,96,3,1,1)
        self.conv6 = nn.Conv2d(96,192,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(192*2*2,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1,192*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[12]:


cnn = rosnet_11_3()
train_test(cnn,0.001,10)


# In[ ]:


class rosnet_11_3x(nn.Module):
    def __init__(self):
        super(rosnet_11_3x,self).__init__()
        self.conv1 = nn.Conv2d(3,6,3,2,1, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(6,12,3,1,1, bias=True)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(12,24,3,1,1, bias=True)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = nn.Conv2d(24,48,3,1,1,bias=True)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.conv5 = nn.Conv2d(48,96,3,1,1,bias=True)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.conv6 = nn.Conv2d(96,192,3,1,1,bias=True)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(192*2*2,256)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = nn.Linear(256,64)
        torch.nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = nn.Linear(64,10)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1,192*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


cnn2 = rosnet_11_3x()
cnn2.cuda
train_test(cnn2,0.001,10)


# In[ ]:


class rosnet_1_2bx(nn.Module):
    def __init__(self):
        super(rosnet_1_2bx,self).__init__()
        self.conv1 = nn.Conv2d(3,8,5,2,1, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8*7*7,256)
        torch.nn.init.xavier_normal(self.fc1.weight)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,10)
        torch.nn.init.xavier_normal(self.fc2.weight)
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))
        x = x.view(-1,8*7*7)
        x = self.fc1(x)
        x = self.fc2(F.relu(self.bn2(x)))
        return x


# In[ ]:


cnn3 = rosnet_1_2bx()

