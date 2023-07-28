#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained= True)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img)

def predict(image_path, model, category_name, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    img = img.to(device)
    model = model.to(device)
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
    
    with torch.no_grad():
        model.eval()
        logps = model(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        top_p, top_class = top_p[0].tolist(), top_class[0].tolist()          
           
        top_classes = [model.idx_to_class[idx] for idx in top_class]
        pred = cat_to_name[top_classes[0]]
        top_classes_name = [cat_to_name[cl].upper() for cl in top_classes]
        print("....\nPrediction of the flower in the image is {}\n....\nPrediction of top 5 classes category names are: {}".format(pred.upper(), top_classes_name))             
    return top_p, top_classes
