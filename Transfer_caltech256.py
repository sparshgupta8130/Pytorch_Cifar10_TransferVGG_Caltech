
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

import cv2
import os
import copy


# In[2]:


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
train_batch=4
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


# In[3]:


def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))


# In[4]:


dataiter = iter(train_data)
images,_ = dataiter.next()
imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[5]:


dataiter2 = iter(valid_data)
images2,labels2 = dataiter2.next()
print(labels2)
imshow(torchvision.utils.make_grid(images2))


# In[6]:


vgg16 = models.vgg16(pretrained=True)
st_dict = vgg16.state_dict()
dtype = torch.FloatTensor


# In[7]:


import torch.nn as nn
class vgg_16_custom(nn.Module):
    def __init__(self):
        super(vgg_16_custom,self).__init__()
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
        
        self.conv11 = nn.Conv2d(512,512,3,padding=1) #24
        self.conv11.weight = torch.nn.Parameter(st_dict["features.24.weight"].type(dtype),requires_grad = False)
        self.conv11.bias  = torch.nn.Parameter(st_dict["features.24.bias"].type(dtype),requires_grad = False)
        
        self.conv12 = nn.Conv2d(512,512,3,padding=1) #26
        self.conv12.weight = torch.nn.Parameter(st_dict["features.26.weight"].type(dtype),requires_grad = False)
        self.conv12.bias  = torch.nn.Parameter(st_dict["features.26.bias"].type(dtype),requires_grad = False)
        
        self.conv13 = nn.Conv2d(512,512,3,padding=1) #28
        self.conv13.weight = torch.nn.Parameter(st_dict["features.28.weight"].type(dtype),requires_grad = False)
        self.conv13.bias  = torch.nn.Parameter(st_dict["features.28.bias"].type(dtype),requires_grad = False)
        
        self.fc1 = nn.Linear(25088,4096)
        self.fc1.weight = torch.nn.Parameter(st_dict["classifier.0.weight"].type(dtype),requires_grad = False)
        self.fc1.bias = torch.nn.Parameter(st_dict["classifier.0.bias"].type(dtype),requires_grad = False)
        
        self.fc2 = nn.Linear(4096,4096)
        self.fc2.weight = torch.nn.Parameter(st_dict["classifier.3.weight"].type(dtype),requires_grad = False)
        self.fc2.bias = torch.nn.Parameter(st_dict["classifier.3.bias"].type(dtype),requires_grad = False)
        
        self.fc3 = nn.Linear(4096,256)
        torch.nn.init.xavier_normal(self.fc3.weight)
        
        self.do = nn.Dropout(0.5)
        
        
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
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(F.relu(self.conv13(x)))
        
        x = x.view(-1,25088)
        
        x = self.do(F.relu(self.fc1(x)))
        x = self.do(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def forward_pass(self,layer,x):
        if layer == "conv1":
            x = self.conv1(x)
            return x
        elif layer == "conv2":
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return x
        elif layer == "conv3":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.conv3(x)
            return x
        elif layer == "conv4":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.conv4(x)
            return x
        elif layer == "conv5":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.conv5(x)
            return x
        elif layer == "conv6":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = self.conv6(x)
            return x
        elif layer == "conv7":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.conv7(x)
            return x
        elif layer == "conv8":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool(F.relu(self.conv7(x)))
            x = self.conv8(x)
            return x
        elif layer == "conv9":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool(F.relu(self.conv7(x)))
            x = F.relu(self.conv8(x))
            x = self.conv9(x)
            return x
        elif layer == "conv10":
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = self.pool(F.relu(self.conv7(x)))
            x = F.relu(self.conv8(x))
            x = F.relu(self.conv9(x))
            x = self.conv10(x)
            return x
        elif layer == "conv11":
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
            x = self.conv11(x)
            return x
        elif layer == "conv12":
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
            x = F.relu(self.conv11(x))
            x = self.conv12(x)
            return x
        elif layer == "conv13":
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
            x = F.relu(self.conv11(x))
            x = F.relu(self.conv12(x))
            x = self.conv13(x)
            return x


# In[8]:


def train_test(cnn,eta=0.001,ep=10):
    cnn.cuda()
    loss = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.fc3.parameters(),lr=eta)
    eps = []
    tr_loss = []
    te_loss = []
    tr_acc = []
    te_acc = []
    
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
        
        temp = 0
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
            l = loss(preds,labels)
            temp += l.data[0]
        tr_loss.append(temp/len(train_idx))
        
        temp = 0
        for i,data in enumerate(test_data,0):
            inputs,labels = data
            inputs = Variable(inputs.cuda())
            labels = labels-1
            #print(len(labels))
            #print(labels)
            labels = labels.view(len(labels))
            labels = Variable(labels.type(torch.LongTensor).cuda())
            optimiser.zero_grad()

            preds = cnn(inputs)
            l = loss(preds,labels)
            temp += l.data[0]
        te_loss.append(temp/8*256)
        
        correct = 0
        total = 0
        for data in train_data:
            images, labels = data
            labels = labels-1
            outputs = cnn(Variable(images.cuda()))
            labels = labels.view(len(labels))
            #labels = Variable(labels.type(torch.LongTensor).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cuda().size(0)
            correct += (predicted == labels.type(torch.LongTensor).cuda()).sum()
        tr_acc.append(100*correct/total)
        
        correct = 0
        total = 0
        for data in test_data:
            images, labels = data
            labels = labels-1
            outputs = cnn(Variable(images.cuda()))
            labels = labels.view(len(labels))
            #labels = Variable(labels.type(torch.LongTensor).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.cuda().size(0)
            correct += (predicted == labels.type(torch.LongTensor).cuda()).sum()
        te_acc.append(100*correct/total)
        eps.append(epoch)
        
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
    
    return eps,tr_loss,te_loss,tr_acc,te_acc


# In[9]:


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


# In[10]:


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model.cuda()
        #self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def visualise_layer(self,epochs=51):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image)
        self.processed_image.cuda()
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        #optimizer = torch.optim.Adam([self.processed_image], lr=5)
        optimizer = torch.optim.SGD([self.processed_image], lr=5,weight_decay=1e-6)
        for i in range(1, epochs):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image.cuda()
            #forward pass layer by layer
            x = self.model.forward_pass(self.selected_layer,x)
#             for index, layer in enumerate(self.model):
#                 # Forward pass layer by layer
#                 x = layer(x)
#                 if index == self.selected_layer:
#                     # Only need to forward until the selected layer is reached
#                     # Now, x is the output of the selected layer
#                     break
            
            
            self.conv_output = x[0, self.selected_filter]
            
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.9f}".format(loss.data[0]))
            
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 5 == 0:
                cv2.imwrite('./generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                            self.created_image)


# In[11]:


cnn = vgg_16_custom()
eps,tr_loss,te_loss,tr_acc,te_acc = train_test(cnn,0.0001,50)

plt.plot(eps,tr_loss)
plt.plot(eps,te_loss)
plt.grid(True)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['Train Loss','Test Loss'])
plt.show()

plt.plot(eps,tr_acc)
plt.plot(eps,te_acc)
plt.grid(True)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train Acc','Test Acc'],loc='lower right')
plt.show()


# In[36]:


cnn_layer = "conv1"
filter_pos = 5
layer_vis = CNNLayerVisualization(cnn_, cnn_layer, filter_pos)
layer_vis.visualise_layer(epochs=500)

