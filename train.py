import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from trainer import load_data, build_model, train_model, test_model, save_checkpoint, select_in_features
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Your data directory.')
    parser.add_argument('--epochs', type=int, help='Number of epochs.')
    parser.add_argument('--arch', type=str, help='model arch please!')
    parser.add_argument("--save_to", type=str, default="checkpoint.pth", help="Checkpoint path.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model.')
    parser.add_argument('--gpu', action='store_true', help='GPU')

    args = parser.parse_args()   
    return args

def main_train():
    #argument from user
    args = get_arguments()
    
    arch = args.arch
    model = build_model(arch)
    learning_rate = args.learning_rate
    epochs = args.epochs
    data_dir = args.data_dir
    save_to = args.save_to
    in_features = select_in_features(arch)
    
    device = torch.device('cuda')
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_datasets, dataloaders = load_data(data_dir)
    #train model
    model = train_model(model, epochs, dataloaders, learning_rate, in_features, device)
    
    #test model
    tested_model = test_model(model, dataloaders, device)
    
    #save checkpoint    
    save_checkpoint(model, image_datasets, epochs, save_to, learning_rate, arch)
    print("You made it so FAR... Model saved!")

    
if __name__ == "__main__":
    main_train()