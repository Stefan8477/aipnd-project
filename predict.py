# Imports here
import torch
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='TestPic.jpg', help='Image to predict')
parser.add_argument('--checkpoint', type=str, default='flower102_checkpoint_CL.pth', help='Model checkpoint for predicting')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

args = parser.parse_args()

# Use command line values when specified
image = args.image   
catfile=args.cat_to_name
checkpoint=args.checkpoint
topk=args.topk
if args.gpu==True:
    device = 'gpu'
else:    
    device='cpu' 


# load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16()
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

modelLOAD=load_checkpoint(checkpoint)

def process_image(image):
    size= 256
    crop = 224
    
    # Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([
        transforms.Resize(size), 
        transforms.CenterCrop(crop), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    numpy_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_image = (np.transpose(numpy_image, (1, 2, 0)) - mean)/std   
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
            
    return numpy_image

def predict(image, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    image=  Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) #for VGG
    if device=='gpu'and torch.cuda.is_available():
        image = image.cuda()
        model.to('cuda')
    result = model(image).topk(topk)
    probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
    classes = result[1].data.cpu().numpy()[0]
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    topk_class = [idx_to_class[x] for x in classes]  
    return probs, topk_class

#ima = process_image(args.image)
ima = process_image(image)
probs, classes = predict(ima, modelLOAD, topk)

print(probs)
print(classes)

#print([cat_to_name[x] for x in classes])
with open(catfile, 'r') as f:
    cat_to_name = json.load(f)

flowernames=[]
for k in range(len(classes)):
    flowername=(cat_to_name.get(classes[k]))
    flowernames.append(flowername)
print(flowernames)

