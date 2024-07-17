import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


#loading data

data_dir = "flowers"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#transforms for the training, validation, testing

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

#dataloaders

train_loader = torch.utils.data.DataLoader(training_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=4)

valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                           batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset,
                                          batch_size=16)

model = models.vgg16(pretrained=True)
print(model)

#frezzing pretrained model parameters to avoid backpop through them

for parameter in model.parameters():
    parameter.requires_grad=False

from collections import OrderedDict

#bulid custom classifier instead of model's own classifier (eikhane model er shes ongshe jei classifier ache take replace kore nijer data related classifier build korbo
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                      ('relu', nn.ReLU()),
                                      ('drop', nn.Dropout(p=0.5)),
                                      ('fc2', nn.Linear(5000, 6)),  #6 as I have only six classes or flowers
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
model.classifier = classifier

#function for validation pass

def validation(model, validateloader, criterion):

    val_loss = 0
    accuracy = 0

    for images, labels in iter(validateloader):

        #images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        probabilities = torch.exp(output)
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy

#loss function and gradient descent

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#train the classifier
def train_classifier():
    epochs = 2
    steps = 0
    print_every = 10

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()

            if steps % print_every == 0:
                model.eval()
                #turn-off gd for validation, svaes memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, valid_loader, criterion)

                print("Epoch : {}/{}.. ".format(e+1, epochs),
                      "training Loss : {:.3f}.. ".format(running_loss/print_every),
                      "validation Loss : {:.3f}.. ".format(validation_loss/len(valid_loader)),
                      "validation Accuracy : {:.3f}.. ".format(accuracy / len(valid_loader))
                      )
                running_loss = 0
                model.train()


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    train_classifier()