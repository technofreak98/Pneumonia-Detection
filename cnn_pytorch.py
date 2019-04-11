import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os 
import cv2
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.nn.functional as F

path1 = "chest_xray/train/NORMAL"
path2 = "chest_xray/train/PNEUMONIA"
path3 = "chest_xray/test/NORMAL"
path4 = "chest_xray/test/PNEUMONIA"

train_data = []
train_label = []
test_data = []
test_label=[]

# count1 = 0
for files in os.listdir(path1):
    
    if (files.endswith(".jpeg")):
        img = cv2.imread(os.path.join(path1,files))
        img = cv2.resize(img, (128, 128)) 
        train_data.append(img)
        train_label.append(0)
#         count1 += 1

# count2 = 0
for files in os.listdir(path2):
    
    if (files.endswith(".jpeg")):
        img = cv2.imread(os.path.join(path2,files))
        img = cv2.resize(img, (128, 128)) 
        train_data.append(img)
        train_label.append(1)
#         count2 += 1

# count1 = 0
for files in os.listdir(path3):
    
    if (files.endswith(".jpeg")):
        img = cv2.imread(os.path.join(path3,files))
        img = cv2.resize(img, (128, 128)) 
        test_data.append(img)
        test_label.append(0)
#         count1 += 1

# count2 = 0
for files in os.listdir(path4):
    
    if (files.endswith(".jpeg")):
        img = cv2.imread(os.path.join(path4,files))
        img = cv2.resize(img, (128, 128)) 
        test_data.append(img)
        test_label.append(1)
#         count2 += 1


from sklearn.utils import shuffle
train_data,train_label = shuffle(train_data,train_label,random_state = 0)
test_data,test_label = shuffle(test_data,test_label,random_state=1)

train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_label = np.asarray(train_label)
test_label = np.asarray(test_label)

train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()
train_label = torch.from_numpy(train_label).float()
test_label = torch.from_numpy(test_label).float()

# print(train_data.shape)
# train_data = train_data.permute((3,1,2))
# test_data = test_data.permute(3,1,2)

train_data.shape
test_data.shape

train_data = train_data.permute((0,3,1,2))
test_data = test_data.permute(0,3,1,2)
train_data.shape

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 2
batch_size = 128
learning_rate = 0.001

# Data loader

train = data_utils.TensorDataset(train_data, train_label)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(test_data, test_label)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

classes = ('Parasitized', 'Uninfected')

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(4*4*512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
    def forward(self, x):
#         print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
#         print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc3(out)
        return out

model = ConvNet(num_classes).to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
correct = 0
count=0
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
#         if outputs == labels:
#             correct +=1
#             count+=1
#         print(loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#                 # print statistics
#         running_loss += loss.item()
#         if (i+1) % 10 == 0:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss))
# #             running_loss = 0.0
#         if (i+1) % 100 == 0:
#             acc = (correct/count)*100     
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#             print("Train accuracy",acc)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
print('Finished Training')

