
# coding: utf-8

# In[1]:


import torch

import torchvision
from torchvision import transforms, models,datasets

from caltech256 import Caltech256

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[14]:


from caltech256 import Caltech256

example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)
        
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=True)
caltech256_val = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=True)
caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=False)

num_train = len(caltech256_train)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))
shuffle =True
if shuffle == True:
    np.random.seed()
    np.random.shuffle(indices)
train_batch=64
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_data = torch.utils.data.DataLoader(
    dataset = caltech256_train, 
    batch_size=train_batch, sampler=train_sampler, 
    num_workers=4)

valid_data = torch.utils.data.DataLoader(
    dataset = caltech256_val, 
    sampler=valid_sampler, 
    num_workers=4)

test_data = torch.utils.data.DataLoader(
    dataset = caltech256_test,
    shuffle = True,
    num_workers = 4)


# In[5]:


def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))


# In[6]:


dataiter = iter(train_data)
images,_ = dataiter.next()
imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[7]:


dataiter2 = iter(valid_data)
images2,labels2 = dataiter2.next()
print(labels2)
imshow(torchvision.utils.make_grid(images2))


# In[8]:


vgg16 = models.vgg16(pretrained=True)
st_dict = vgg16.state_dict()
dtype = torch.FloatTensor


# In[9]:


vgg16.eval()


# In[10]:


import torch.nn as nn
class vgg_16_probe1(nn.Module):
    def __init__(self):
        super(vgg_16_probe1,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv1.weight = torch.nn.Parameter(st_dict["features.0.weight"].type(dtype),requires_grad = False)
        self.conv1.bias  = torch.nn.Parameter(st_dict["features.0.bias"].type(dtype),requires_grad = False)
        
        self.conv2 = nn.Conv2d(64,64,3,padding =1)
        self.conv2.weight = torch.nn.Parameter(st_dict["features.2.weight"].type(dtype),requires_grad = False)
        self.conv2.bias  = torch.nn.Parameter(st_dict["features.2.bias"].type(dtype),requires_grad = False)
        
        self.pool = nn.MaxPool2d(2,2) #4
        
        self.conv3 = nn.Conv2d(64,128,3,padding=1) #5
        self.conv3.weight = torch.nn.Parameter(st_dict["features.5.weight"].type(dtype),requires_grad = False)
        self.conv3.bias  = torch.nn.Parameter(st_dict["features.5.bias"].type(dtype),requires_grad = False)
        
        self.conv4 = nn.Conv2d(128,128,3,padding=1) #7
        self.conv4.weight = torch.nn.Parameter(st_dict["features.7.weight"].type(dtype),requires_grad = False)
        self.conv4.bias  = torch.nn.Parameter(st_dict["features.7.bias"].type(dtype),requires_grad = False)
        
        self.conv5 = nn.Conv2d(128,256,3,padding=1) #10
        self.conv5.weight = torch.nn.Parameter(st_dict["features.10.weight"].type(dtype),requires_grad = False)
        self.conv5.bias  = torch.nn.Parameter(st_dict["features.10.bias"].type(dtype),requires_grad = False)
        
        self.conv6 = nn.Conv2d(256,256,3,padding=1) #12
        self.conv6.weight = torch.nn.Parameter(st_dict["features.12.weight"].type(dtype),requires_grad = False)
        self.conv6.bias  = torch.nn.Parameter(st_dict["features.12.bias"].type(dtype),requires_grad = False)
        
        self.conv7 = nn.Conv2d(256,256,3,padding=1) #14
        self.conv7.weight = torch.nn.Parameter(st_dict["features.14.weight"].type(dtype),requires_grad = False)
        self.conv7.bias  = torch.nn.Parameter(st_dict["features.14.bias"].type(dtype),requires_grad = False)
        
#         self.conv8 = nn.Conv2d(256,512,3,padding=1) #17
#         self.conv8.weight = torch.nn.Parameter(st_dict["features.17.weight"].type(dtype),requires_grad = False)
#         self.conv8.bias  = torch.nn.Parameter(st_dict["features.17.bias"].type(dtype),requires_grad = False)
        
#         self.conv9 = nn.Conv2d(512,512,3,padding=1) #19
#         self.conv9.weight = torch.nn.Parameter(st_dict["features.19.weight"].type(dtype),requires_grad = False)
#         self.conv9.bias  = torch.nn.Parameter(st_dict["features.19.bias"].type(dtype),requires_grad = False)
        
#         self.conv10 = nn.Conv2d(512,512,3,padding=1) #21
#         self.conv10.weight = torch.nn.Parameter(st_dict["features.21.weight"].type(dtype),requires_grad = False)
#         self.conv10.bias  = torch.nn.Parameter(st_dict["features.21.bias"].type(dtype),requires_grad = False)
        
#         self.conv11 = nn.Conv2d(512,512,3,padding=1) #24
#         self.conv11.weight = torch.nn.Parameter(st_dict["features.24.weight"].type(dtype),requires_grad = False)
#         self.conv11.bias  = torch.nn.Parameter(st_dict["features.24.bias"].type(dtype),requires_grad = False)
        
#         self.conv12 = nn.Conv2d(512,512,3,padding=1) #26
#         self.conv12.weight = torch.nn.Parameter(st_dict["features.26.weight"].type(dtype),requires_grad = False)
#         self.conv12.bias  = torch.nn.Parameter(st_dict["features.26.bias"].type(dtype),requires_grad = False)
        
#         self.conv13 = nn.Conv2d(512,512,3,padding=1) #28
#         self.conv13.weight = torch.nn.Parameter(st_dict["features.28.weight"].type(dtype),requires_grad = False)
#         self.conv13.bias  = torch.nn.Parameter(st_dict["features.28.bias"].type(dtype),requires_grad = False)
        
#         self.fc1 = nn.Linear(25088,4096)
#         self.fc1.weight = torch.nn.Parameter(st_dict["classifier.0.weight"].type(dtype),requires_grad = False)
#         self.fc1.bias = torch.nn.Parameter(st_dict["classifier.0.bias"].type(dtype),requires_grad = False)
        
        self.fc2 = nn.Linear(200704,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
        #self.fc2.weight = torch.nn.Parameter(st_dict["classifier.3.weight"].type(dtype),requires_grad = False)
        #self.fc2.bias = torch.nn.Parameter(st_dict["classifier.3.bias"].type(dtype),requires_grad = False)
        
        self.fc3 = nn.Linear(1024,256)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
#         self.do = nn.Dropout(0.5)
        
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
#         x = F.relu(self.conv8(x))
#         x = F.relu(self.conv9(x))
#         x = self.pool(F.relu(self.conv10(x)))
#         x = F.relu(self.conv11(x))
#         x = F.relu(self.conv12(x))
#         x = self.pool(F.relu(self.conv13(x)))
        
        x = x.view(-1,200704)
        
#         x = self.do(F.relu(self.fc1(x)))
#         x = self.do(F.relu(self.fc2(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# In[11]:


import torch.nn as nn
class vgg_16_probe2(nn.Module):
    def __init__(self):
        super(vgg_16_probe2,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv1.weight = torch.nn.Parameter(st_dict["features.0.weight"].type(dtype),requires_grad = False)
        self.conv1.bias  = torch.nn.Parameter(st_dict["features.0.bias"].type(dtype),requires_grad = False)
        
        self.conv2 = nn.Conv2d(64,64,3,padding =1)
        self.conv2.weight = torch.nn.Parameter(st_dict["features.2.weight"].type(dtype),requires_grad = False)
        self.conv2.bias  = torch.nn.Parameter(st_dict["features.2.bias"].type(dtype),requires_grad = False)
        
        self.pool = nn.MaxPool2d(2,2) #4
        
        self.conv3 = nn.Conv2d(64,128,3,padding=1) #5
        self.conv3.weight = torch.nn.Parameter(st_dict["features.5.weight"].type(dtype),requires_grad = False)
        self.conv3.bias  = torch.nn.Parameter(st_dict["features.5.bias"].type(dtype),requires_grad = False)
        
        self.conv4 = nn.Conv2d(128,128,3,padding=1) #7
        self.conv4.weight = torch.nn.Parameter(st_dict["features.7.weight"].type(dtype),requires_grad = False)
        self.conv4.bias  = torch.nn.Parameter(st_dict["features.7.bias"].type(dtype),requires_grad = False)
        
        self.conv5 = nn.Conv2d(128,256,3,padding=1) #10
        self.conv5.weight = torch.nn.Parameter(st_dict["features.10.weight"].type(dtype),requires_grad = False)
        self.conv5.bias  = torch.nn.Parameter(st_dict["features.10.bias"].type(dtype),requires_grad = False)
        
        self.conv6 = nn.Conv2d(256,256,3,padding=1) #12
        self.conv6.weight = torch.nn.Parameter(st_dict["features.12.weight"].type(dtype),requires_grad = False)
        self.conv6.bias  = torch.nn.Parameter(st_dict["features.12.bias"].type(dtype),requires_grad = False)
        
        self.conv7 = nn.Conv2d(256,256,3,padding=1) #14
        self.conv7.weight = torch.nn.Parameter(st_dict["features.14.weight"].type(dtype),requires_grad = False)
        self.conv7.bias  = torch.nn.Parameter(st_dict["features.14.bias"].type(dtype),requires_grad = False)
        
        self.conv8 = nn.Conv2d(256,512,3,padding=1) #17
        self.conv8.weight = torch.nn.Parameter(st_dict["features.17.weight"].type(dtype),requires_grad = False)
        self.conv8.bias  = torch.nn.Parameter(st_dict["features.17.bias"].type(dtype),requires_grad = False)
        
        self.conv9 = nn.Conv2d(512,512,3,padding=1) #19
        self.conv9.weight = torch.nn.Parameter(st_dict["features.19.weight"].type(dtype),requires_grad = False)
        self.conv9.bias  = torch.nn.Parameter(st_dict["features.19.bias"].type(dtype),requires_grad = False)
        
        self.conv10 = nn.Conv2d(512,512,3,padding=1) #21
        self.conv10.weight = torch.nn.Parameter(st_dict["features.21.weight"].type(dtype),requires_grad = False)
        self.conv10.bias  = torch.nn.Parameter(st_dict["features.21.bias"].type(dtype),requires_grad = False)
        
#         self.conv11 = nn.Conv2d(512,512,3,padding=1) #24
#         self.conv11.weight = torch.nn.Parameter(st_dict["features.24.weight"].type(dtype),requires_grad = False)
#         self.conv11.bias  = torch.nn.Parameter(st_dict["features.24.bias"].type(dtype),requires_grad = False)
        
#         self.conv12 = nn.Conv2d(512,512,3,padding=1) #26
#         self.conv12.weight = torch.nn.Parameter(st_dict["features.26.weight"].type(dtype),requires_grad = False)
#         self.conv12.bias  = torch.nn.Parameter(st_dict["features.26.bias"].type(dtype),requires_grad = False)
        
#         self.conv13 = nn.Conv2d(512,512,3,padding=1) #28
#         self.conv13.weight = torch.nn.Parameter(st_dict["features.28.weight"].type(dtype),requires_grad = False)
#         self.conv13.bias  = torch.nn.Parameter(st_dict["features.28.bias"].type(dtype),requires_grad = False)
        
#         self.fc1 = nn.Linear(25088,4096)
#         self.fc1.weight = torch.nn.Parameter(st_dict["classifier.0.weight"].type(dtype),requires_grad = False)
#         self.fc1.bias = torch.nn.Parameter(st_dict["classifier.0.bias"].type(dtype),requires_grad = False)
        
        self.fc2 = nn.Linear(100352,1024)
        torch.nn.init.xavier_normal(self.fc2.weight)
#         self.fc2.weight = torch.nn.Parameter(st_dict["classifier.3.weight"].type(dtype),requires_grad = False)
#         self.fc2.bias = torch.nn.Parameter(st_dict["classifier.3.bias"].type(dtype),requires_grad = False)
        
        self.fc3 = nn.Linear(1024,256)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
#         self.do = nn.Dropout(0.5)
        
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(F.relu(self.conv10(x)))
#         x = F.relu(self.conv11(x))
#         x = F.relu(self.conv12(x))
#         x = self.pool(F.relu(self.conv13(x)))
        
        x = x.view(-1,100352)
        
#         x = self.do(F.relu(self.fc1(x)))
#         x = self.do(F.relu(self.fc2(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# In[17]:


def train_test(cnn,eta=0.001,ep=10):
    cnn.cuda()
    loss = nn.CrossEntropyLoss()
    optimiser = optim.SGD([{'params':cnn.fc2.parameters()},{'params':cnn.fc3.parameters()}],lr=eta,momentum=0.9,nesterov=True)

    for epoch in range(ep):

        l_tr= 0.0
        for i,data in enumerate(train_data,0):

            inputs,labels = data
            inputs = Variable(inputs.cuda())
            labels = labels-1
            #print(len(labels))
            #print(labels)
            labels = labels.view(len(labels))
            labels = Variable(labels.type(torch.LongTensor).cuda())
            optimiser.zero_grad()

            preds = cnn(inputs)
            loss_tr = loss(preds,labels)
            loss_tr.backward()
            optimiser.step()

            l_tr += loss_tr.data[0]
            #print(i)
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, l_tr / 100))
                l_tr = 0.0
    print ('finished training')
    
    correct = 0
    total = 0
    for data in test_data:
        images, labels = data
        labels = labels-1
        outputs = cnn(Variable(images.cuda()))
        labels = labels.view(1)
        #labels = Variable(labels.type(torch.LongTensor).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        correct += (predicted == labels.type(torch.LongTensor).cuda()).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     for data in test_data:
#         images, labels = data
#         labels = labels-1
#         outputs = cnn(Variable(images.cuda()))
#         labels = labels.view(1)
#         #labels = Variable(labels.type(torch.LongTensor).cuda())
#         _, predicted = torch.max(outputs.data, 1)
#         c = (predicted == labels.type(torch.LongTensor).cuda()).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i]
#             class_total[label] += 1


#     for i in range(10):
#         print('Accuracy of %2d : %2d %%' % (
#              i+1 ,100 * class_correct[i] / class_total[i]))


# In[19]:


cnn = vgg_16_probe1()
train_test(cnn,0.001,50)


# In[18]:


cnn_ = vgg_16_probe2()
train_test(cnn_,0.001,50)

