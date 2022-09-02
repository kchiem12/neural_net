from ast import Str
from importlib.resources import path
import torch
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customDataset import RailroadsDataset
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import time

# set the device to run it on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparamets
input_size = 150528
learning_rate = 0.001
batch_size = 30
num_epochs = 12

#load data
training_railroads_data = RailroadsDataset(csv="./railroads_dataset/railroads_train.csv", path_dir='railroads_dataset', transform=transforms.ToTensor())
testing_railroads_data = RailroadsDataset(csv="./railroads_dataset/railroads_test.csv", path_dir='railroads_dataset', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_railroads_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testing_railroads_data, batch_size=batch_size, shuffle=True)

# the model
class NeuralNetwork(nn.Module):
    def __init__(self, num_channels, num_classes): 
        super(NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=7, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # the  stride is to reduce the spatial size
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features= 34992, out_features= 500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=0)

    def forward(self, input):

        # pass input through first layer of conv
        output = F.relu(self.maxpool(self.conv1(input)))

        # pass through second layer
        output = F.relu(self.maxpool(self.conv2(output))) 

        # print(output.shape)
        # print('\n\n --------------------------------------- ')

        # flatten the tensor to one dimension
        output = output.view(output.size(0), -1) 
        output = self.fc1(output)
        output = torch.log_softmax(output, dim=1)        

        return output


# instantiate neural net
model = NeuralNetwork(num_channels=3, num_classes=2)

# defining the loss function
loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr= 0.001, weight_decay=0.0001) # weight decay helps with more accurate descent


# To save the model
def saveModel():
    path = "./railroad_tracks_nn"
    torch.save(model.state_dict(), path)

# Function to test model
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #run model on test set
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy+= (predicted==labels).sum().item()

    accuracy = (100*accuracy) / total
    return accuracy
    
# Training function
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # To keep track of loss/accuracy
    history = {
        "train_loss": [],
        "train_acc": []
    }

    # begin the timer !!!
    print("Beginning training")
    startTime = time.time()

    bestAcc = 0


    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # setting model into training mode
        model.train()

        # total loss
        totalTrainLoss = 0

        # number of training images correct
        trainCorrect = 0

        for i, (images, labels) in enumerate(train_loader):
            
            # get the inputs and send to device
            images = images.to(device)
            labels = labels.to(device)

            # get the prediction
            preds = model(images)

            losss = loss(preds, labels)

            # ZERO GRAD (ALWAYS REMEMBER TO DO THIS !!!!)
            # then back propagate and optimize
            optimizer.zero_grad()
            losss.backward()
            optimizer.step()

            totalTrainLoss += losss
            trainCorrect += (preds.argmax(1) == labels).type(torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / 10
        trainCorrect = trainCorrect / len(train_loader.dataset)
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["train_acc"].append(trainCorrect)

        print(f'EPOCH {epoch + 1}/{num_epochs}')
        print(f'Train loss: {avgTrainLoss}, Train accuracy: {trainCorrect}')
        
        if trainCorrect > best_accuracy:
            bestAccuracy = trainCorrect
            saveModel()
    
    endTime = time.time()

    print(f'TOTAL TIME TO TRAIN MODEL: {endTime - startTime}')

if __name__ == "__main__":
    train(num_epochs= num_epochs)
    print(f'\n\nThe accuracy is {testAccuracy()}')
    saveModel()
    



















# from importlib.resources import path
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision
# from torch.utils.data import DataLoader
# from customDataset import RailroadsDataset
# from torch.optim import Adam
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from PIL import Image

# # set the device to run it on
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #hyperparamets
# input_size = 50176
# num_classes = 2
# learning_rate = 0.001
# batch_size = 30
# num_epochs = 5

# #load data
# training_railroads_data = RailroadsDataset(csv="./railroads_dataset/railroads_train.csv", path_dir='railroads_dataset', transform=transforms.ToTensor())
# testing_railroads_data = RailroadsDataset(csv="./railroads_dataset/railroads_test.csv", path_dir='railroads_dataset', transform=transforms.ToTensor())

# train_loader = DataLoader(dataset=training_railroads_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=testing_railroads_data, batch_size=batch_size, shuffle=True)

# # the model
# class NeuralNetwork(nn.Module):
#     def __init__(self): 
#         super(NeuralNetwork, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(12)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#         self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(24)
#         self.fc1 = nn.Linear(269664, 2)

#     def forward(self, input):
#         output = F.relu(self.bn1(self.conv1(input)))      
#         output = F.relu(self.bn2(self.conv2(output)))     
#         output = self.pool(output)                        
#         output = F.relu(self.bn4(self.conv4(output)))     
#         output = F.relu(self.bn5(self.conv5(output)))     
#         output = output.view(-1, 24*106*106)
#         output = self.fc1(output)

#         return output

# # instantiate neural net
# model = NeuralNetwork()

# # defining the loss function
# loss = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr= 0.001, weight_decay=0.0001)


# # To save the model
# def saveModel():
#     path = "./FirstModel.pth"
#     torch.save(model.state_dict(), path)

# # Function to test model
# def testAccuracy():
#     model.eval()
#     accuracy = 0.0
#     total = 0.0

#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             #run model on test set
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             accuracy+= (predicted==labels).sum().item()

#     accuracy = (100*accuracy) / total
#     return accuracy
    
# # Training function
# def train(num_epochs):
    
#     best_accuracy = 0.0

#     # Define your execution device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("The model will be running on", device, "device")
#     # Convert model parameters and buffers to CPU or Cuda
#     model.to(device)

#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#         running_acc = 0.0
#         print("Running this epoch")

#         for i, (images, labels) in enumerate(train_loader):
            
#             # get the inputs
#             images = images.to(device)
#             labels = labels.to(device)
#             print(images)

#             # ZERO GRAD (ALWAYS REMEMBER TO DO THIS !!!!)
#             optimizer.zero_grad()

#             # predict classes using images from the training set
#             outputs = model(images)


#             # compute the loss based on model output and real labels
#             losss = loss(outputs, labels)

#             # backpropagate the loss
#             losss.backward()

#             # adjust parameters based on the calculated gradients
#             optimizer.step()

#             print("The loss for this is ", losss.item())

#             # Let's print statistics for every 25 images
#             running_loss += losss.item()     # extract the loss value
#             if i % 25 == 24:    
#                 # print every 25 times (takes the average loss for every 25 images)
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 25))
#                 # zero the loss
#                 running_loss = 0.0

#         # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
#         accuracy = testAccuracy()
#         print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
#         # we want to save the model if the accuracy is the best
#         if accuracy > best_accuracy:
#             saveModel()
#             best_accuracy = accuracy

# if __name__ == "__main__":
#     train(2)
    