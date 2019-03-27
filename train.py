# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict
import time
import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Path to dataset ')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--arch', type=str, default='vgg', help='architecture [available: densenet, vgg]', required=True)
parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=16, help='Size for a batch')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_units_Layer1', type=int, default=4000, help='hidden units for fc layer 1')
parser.add_argument('--hidden_units_Layer2', type=int, default=630, help='hidden units for fc layer 2')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--checkpoint' , type=str, default='flower102_checkpoint_CL.pth', help='path of your saved model')
args = parser.parse_args()

data_dir = args.data_dir  
lr = args.learning_rate 
epochs = args.epochs
Layer1=args.hidden_units_Layer1
Layer2=args.hidden_units_Layer2
catfile=args.cat_to_name
checkpoint_CL=args.checkpoint
b_size=args.batchsize

if args.arch=='vgg':
    model = models.vgg16(pretrained=True)
    inputlayer=25088
else:    
    model = models.densenet121(pretrained=True)
    inputlayer=1024
if args.gpu==True:
    device = 'gpu'
    print('GPU calculation')
else:    
    device='cpu'
    print('cpu calculation')

# Directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# DONE: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=b_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=b_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=b_size)

# Label mapping
with open(catfile, 'r') as f:
    cat_to_name = json.load(f)
    
#Build and train your network

def do_deep_learning(model, trainloader, print_every, criterion, optimizer, epochs, device='cpu'):
    start = time.time()
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device=='gpu' and torch.cuda.is_available():
        model.to('cuda')
        print('GPU available')
    else:
        print('GPU NOT available')

    for e in range(epochs):
        train_loss = 0
        steps=0
        model.train()
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if device=='gpu'and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
                
            if steps % print_every == 0:
                model.eval()
                valid_loss=0
                valid_steps=0
                valid_correct=0
                with torch.no_grad():
                    for ii, (valid_inputs, valid_labels) in enumerate(validloader):
                        valid_steps += valid_labels.size(0)
                        valid_inputs, valid_labels = valid_inputs.to('cuda'), valid_labels.to('cuda')
                        valid_outputs = model(valid_inputs)
                        valid_loss += criterion(valid_outputs, valid_labels).item()
                        _, preds = torch.max(outputs, 1)
                        valid_correct+= (preds == labels).sum().item()
                    
                    TrainLoss=train_loss/steps
                    ValidLoss=valid_loss/valid_steps
                    ValidAcc=100*valid_correct/valid_steps
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. Validation Loss: {:.3f}.. Validation Accuracy:  {:2.2f}%".format(TrainLoss, ValidLoss, ValidAcc))
                    model.train()
    
    trainingtime = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
    trainingtime // 60, trainingtime % 60))
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputlayer, Layer1)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(Layer1, Layer2)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(Layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
do_deep_learning(model, trainloader, 40, criterion, optimizer, epochs, device)
# validation on the test set
def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device=='gpu'and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
                model.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

check_accuracy_on_test(testloader)

model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': [3, 224, 224],
              'batch_size': trainloader.batch_size,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': epochs
             }
torch.save(checkpoint, 'flower102_checkpoint_CL.pth')