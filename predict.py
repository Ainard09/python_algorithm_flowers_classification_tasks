import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from predict_trainer import load_checkpoint, process_image, predict 
import argparse
import json

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_pth', type=str, help='Path to your image.')
    parser.add_argument("checkpoint", type=str, help='Checkpoint of your model.')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes.')
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help="Display category names.")
    parser.add_argument('--gpu', action='store_true', help='GPU')
    
    args = parser.parse_args()
    return args

def main_predict():
    args = get_argument()
    image = args.image_pth
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name_filepath = args.category_names
    checkpoint = args.checkpoint
    
    #with open(cat_to_name_filepath, 'r') as f:
        #cat_to_name = json.load(f)
    
    device = torch.device("cuda")
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_checkpoint(checkpoint)
    
    # predict top probabilites and classes and show image
    predict(image, model, cat_to_name_filepath, device, top_k)
    
if __name__ == "__main__":
    main_predict()
