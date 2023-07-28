import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from tqdm import tqdm

def load_data(data_dir):
    """ function to load dataset from filepath """
    # define train, validation and test dataset
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # define data transforms for the 3 datasets
    data_transforms = [
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
]

# Load the datasets with ImageFolder
    image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms[0]),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms[1]),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms[2])
}

# Define the dataloaders
    dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
}
    return image_datasets, dataloaders
    
def build_model(arch):
    return getattr(models, arch)(pretrained=True)

def select_in_features(ar):
    """ function to define in feature nodes for the model 
    """
    model_arch = ar
    #check if model architecture entered by the user is within the algorithm production
    if model_arch == 'densenet121':
        in_features = 1024
    elif model_arch == 'vgg16':
        in_features = 25088
    elif model_arch == 'alexnet':
        in_features = 9216
    else:
        print("The algorithm only supports these architectures [densenet121, vgg16, alexnet,]. PLEASE check your entered architecture; {} and try again!".format(model_arch))
    return in_features

def train_model(model, epochs, dataloaders, learning_rate, in_features, device):
    """ train model and validate the model """
    #freeze the gradient descent of the model
    for param in model.parameters():
        param.requires_grad = False
        
    # add fc layers
    classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )

    # add fc to the model
    model.classifier = classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    running_loss = 0
    steps = 0
    print_every = 80

    for e in range(epochs):
        print("Start model training!!!")
        for images, labels in tqdm(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
            steps += 1
            #zero the optimizer
            optimizer.zero_grad()

            #feedforward
            logps = model.forward(images)
            loss = criterion(logps, labels)
            running_loss += loss.item()

            #backpropagate and update the optimizer
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    #iterate through the validation dataloader
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)

                        #model in eveluation mode 
                        model.eval()

                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        #calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every :.3f}.. "
                      f"Test loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                #back to training mode
                model.train()
                running_loss = 0
    print("Nice training the model, Done!")
    return model

def test_model(model, dataloaders, device):
    print("Testing the accuracy of the model!")
    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0

    with torch.no_grad():     
        #iterate through the validation dataloader
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)

            #model in eveluation mode 
            model.eval()

            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            #calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print("Accuracy: {:.3f}".format(accuracy / len(dataloaders['test'])))
        model.train()
       
def save_checkpoint(model, image_datasets, epochs, save_dir, learning_rate, arch):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    checkpoint = {'classifier' : model.classifier,
             'class_to_idx' : image_datasets['train'].class_to_idx,
             'state_dict' : model.state_dict(),
             'optimizer_state_dict' : optimizer.state_dict(),
             'epochs' : epochs,
              'arch' : arch
}
    torch.save(checkpoint, save_dir)
    
    
    
    
    