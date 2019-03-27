The project is seperated in two parts. Part one is a Jupiter Notebook and you my find the code below. Part two is a command line application. You may find the code for this in "train.py" and "predict.py"
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.


```python
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mplimg

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time

from PIL import Image
from torch.autograd import Variable


import json
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
# DONE: Define your transforms for the training, validation, and testing sets
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



# DONE: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

b_size=16

# DONE: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=b_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=b_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=b_size)
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.


```python
# DONE: Build and train your network

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    start = time.time()
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        train_loss = 0.0 
        steps=0
        model.train()
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
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
                          "Training Loss: {:.3f}.. Validation Loss: {:.3f}.. Validation Accuracy: {:2.2f}%".format(TrainLoss, ValidLoss, ValidAcc))
                    model.train()
                    
    trainingtime = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
    trainingtime // 60, trainingtime % 60))
                    
model = models.vgg16(pretrained=True)
model

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4000, 630)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(630, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

epochs=4

do_deep_learning(model, trainloader, validloader, epochs, 40, criterion, optimizer, 'gpu')

```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:26<00:00, 21058125.31it/s]


    Epoch: 1/4..  Training Loss: 4.400.. Validation Loss: 0.209.. Validation Accuracy: 25.43%
    Epoch: 1/4..  Training Loss: 3.731.. Validation Loss: 0.159.. Validation Accuracy: 50.86%
    Epoch: 1/4..  Training Loss: 3.287.. Validation Loss: 0.116.. Validation Accuracy: 69.93%
    Epoch: 1/4..  Training Loss: 2.958.. Validation Loss: 0.087.. Validation Accuracy: 57.21%
    Epoch: 1/4..  Training Loss: 2.737.. Validation Loss: 0.081.. Validation Accuracy: 50.86%
    Epoch: 1/4..  Training Loss: 2.565.. Validation Loss: 0.081.. Validation Accuracy: 63.57%
    Epoch: 1/4..  Training Loss: 2.427.. Validation Loss: 0.070.. Validation Accuracy: 57.21%
    Epoch: 1/4..  Training Loss: 2.322.. Validation Loss: 0.066.. Validation Accuracy: 63.57%
    Epoch: 1/4..  Training Loss: 2.216.. Validation Loss: 0.061.. Validation Accuracy: 57.21%
    Epoch: 1/4..  Training Loss: 2.141.. Validation Loss: 0.054.. Validation Accuracy: 82.64%
    Epoch: 2/4..  Training Loss: 1.251.. Validation Loss: 0.060.. Validation Accuracy: 69.93%
    Epoch: 2/4..  Training Loss: 1.200.. Validation Loss: 0.054.. Validation Accuracy: 69.93%
    Epoch: 2/4..  Training Loss: 1.242.. Validation Loss: 0.057.. Validation Accuracy: 69.93%
    Epoch: 2/4..  Training Loss: 1.236.. Validation Loss: 0.052.. Validation Accuracy: 76.28%
    Epoch: 2/4..  Training Loss: 1.196.. Validation Loss: 0.050.. Validation Accuracy: 50.86%
    Epoch: 2/4..  Training Loss: 1.182.. Validation Loss: 0.052.. Validation Accuracy: 82.64%
    Epoch: 2/4..  Training Loss: 1.173.. Validation Loss: 0.052.. Validation Accuracy: 76.28%
    Epoch: 2/4..  Training Loss: 1.174.. Validation Loss: 0.048.. Validation Accuracy: 76.28%
    Epoch: 2/4..  Training Loss: 1.163.. Validation Loss: 0.049.. Validation Accuracy: 69.93%
    Epoch: 2/4..  Training Loss: 1.158.. Validation Loss: 0.043.. Validation Accuracy: 69.93%
    Epoch: 3/4..  Training Loss: 0.924.. Validation Loss: 0.039.. Validation Accuracy: 63.57%
    Epoch: 3/4..  Training Loss: 0.902.. Validation Loss: 0.052.. Validation Accuracy: 50.86%
    Epoch: 3/4..  Training Loss: 0.922.. Validation Loss: 0.043.. Validation Accuracy: 69.93%
    Epoch: 3/4..  Training Loss: 0.915.. Validation Loss: 0.044.. Validation Accuracy: 82.64%
    Epoch: 3/4..  Training Loss: 0.952.. Validation Loss: 0.043.. Validation Accuracy: 69.93%
    Epoch: 3/4..  Training Loss: 0.936.. Validation Loss: 0.041.. Validation Accuracy: 95.35%
    Epoch: 3/4..  Training Loss: 0.949.. Validation Loss: 0.046.. Validation Accuracy: 76.28%
    Epoch: 3/4..  Training Loss: 0.954.. Validation Loss: 0.035.. Validation Accuracy: 63.57%
    Epoch: 3/4..  Training Loss: 0.970.. Validation Loss: 0.048.. Validation Accuracy: 63.57%
    Epoch: 3/4..  Training Loss: 0.975.. Validation Loss: 0.061.. Validation Accuracy: 69.93%
    Epoch: 4/4..  Training Loss: 0.800.. Validation Loss: 0.043.. Validation Accuracy: 69.93%
    Epoch: 4/4..  Training Loss: 0.866.. Validation Loss: 0.039.. Validation Accuracy: 101.71%
    Epoch: 4/4..  Training Loss: 0.866.. Validation Loss: 0.037.. Validation Accuracy: 89.00%
    Epoch: 4/4..  Training Loss: 0.857.. Validation Loss: 0.051.. Validation Accuracy: 82.64%
    Epoch: 4/4..  Training Loss: 0.878.. Validation Loss: 0.038.. Validation Accuracy: 76.28%
    Epoch: 4/4..  Training Loss: 0.883.. Validation Loss: 0.036.. Validation Accuracy: 82.64%
    Epoch: 4/4..  Training Loss: 0.879.. Validation Loss: 0.038.. Validation Accuracy: 57.21%
    Epoch: 4/4..  Training Loss: 0.886.. Validation Loss: 0.044.. Validation Accuracy: 76.28%
    Epoch: 4/4..  Training Loss: 0.887.. Validation Loss: 0.033.. Validation Accuracy: 82.64%
    Epoch: 4/4..  Training Loss: 0.891.. Validation Loss: 0.038.. Validation Accuracy: 82.64%
    Training complete in 30m 31s


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# DONE: Do validation on the test set
def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {:2.2f}%' .format((100 * correct / total)))

check_accuracy_on_test(testloader)
```

    Accuracy of the network on the 10000 test images: 81.44%


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# DONE: Save the checkpoint 

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
torch.save(checkpoint, 'flower102_checkpoint.pth')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
# DONE: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16()

    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

modelLOAD=load_checkpoint('flower102_checkpoint.pth')
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256
    crop = 224
    border= (size-crop)/2
    area=(border, border, size-border, size-border)
    
    # DONE: Process a PIL image for use in a PyTorch model

    img_loader = transforms.Compose([
        transforms.Resize(size), 
        transforms.CenterCrop(crop), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image)
    nump_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nump_image = (np.transpose(nump_image, (1, 2, 0)) - mean)/std  
    
    #nump_image = (nump_image - mean)/std 
    nump_image = np.transpose(nump_image, (2, 0, 1))
            
    return nump_image
    
ima = process_image('TestPic.jpg')

```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

imshow(ima)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0b63247518>




![png](output_18_1.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    image=  Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) #for VGG
    image = image.cuda()
    model.to('cuda')
    result = model(image).topk(topk)
    probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
    classes = result[1].data.cpu().numpy()[0]
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    topk_class = [idx_to_class[x] for x in classes]  
    return probs, topk_class


probs, classes = predict(ima, modelLOAD)

print(probs)
print(classes)

#print([cat_to_name[x] for x in classes])
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

flowernames=[]

for k in range(len(classes)):
    flowername=(cat_to_name.get(classes[k]))
    flowernames.append(flowername)

print(flowernames)

```

    [  9.99374807e-01   2.69084412e-04   2.68341217e-04   6.20453575e-05
       2.56990552e-05]
    ['12', '41', '50', '77', '71']
    ["colt's foot", 'barbeton daisy', 'common dandelion', 'passion flower', 'gazania']


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# DONE: Display an image along with the top 5 classes
# Display an image along with the top 5 classes
img = mplimg.imread('TestPic.jpg')

f, ax = plt.subplots(2,1,figsize=(10,10))

ax[0].imshow(img)
ax[0].set_title("colt's foot")

y_pos = np.arange(len(classes))

ax[1].barh(y_pos, probs*100, align='center', color='green')
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(flowernames)
ax[1].invert_yaxis()  # labels read top-to-bottom
_ = ax[1].set_xlabel('Probs [%]')
```


![png](output_22_0.png)

