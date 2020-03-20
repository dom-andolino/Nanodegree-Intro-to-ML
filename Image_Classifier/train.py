# Imports here
import numpy as np
import time
import json
import sys
import os
import argparse
from datetime import datetime, date

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image

from workspace_utils import active_session


def get_data_loaders(data_dir):
    '''
    Take data path as input. 
    Perform transforms. Load images. 
    Ouput loaders for train/valid/test data
    '''
    
    #path to train valid and test data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    #same for valid and test dataset
    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return(trainloader, validloader, testloader)


def model_setup(arch, hidden_units, device, data_dir):
    '''
    Get pre-trained transfer model.
    setup classifier
    Output setup model, ready for training. Include in and out feat nums
    '''
    
    
    #get transfer model pretrained based on arch from command line
    model = eval("models."+arch+"(pretrained=True)")
    num_in_feats = model.classifier[0].in_features
    num_out_feats = len(next(os.walk(data_dir + '/train'))[1])
    
    if hidden_units >= num_in_feats or hidden_units <= num_out_feats:
        raise Exception("Hidden units needs to be between {} (the number of input features to the transfer model) and {} (the number of output categories).".format(num_in_feats, num_out_feats))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(num_in_feats, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, num_out_feats),
                                     nn.LogSoftmax(dim=1))

    model.to(device)
    model.train()
    
    return(model, num_in_feats, num_out_feats)


def model_train(model, epochs, criterion, optimizer, trainloader, validloader, device):
    '''
    Train the model, print train loss, valid loss and valid accuracy.
    return trained model and optimizer
    '''
    
    print('\nStarting Training\n')
    model.train()
    
    running_loss = 0
    print_every = 20
    
    with active_session():
        
        for epoch in range(epochs):
            steps = 0
            start_epoch = time.time()
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train Batch {steps}/{len(trainloader)}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

            stop_epoch = time.time()
            epoch_dur = (stop_epoch - start_epoch)/60
            print('Epoch {} duration: {:.1f} mins'.format(epoch+1, epoch_dur))
    
    print('\nTraining Complete\n')
    
    return(model, optimizer)    


def save_model(model, num_in_feats, num_out_feats, args, optimizer):
    '''
    save train model to a checkpoint
    '''
    
    checkpoint = {'input_size': num_in_feats,
                  'output_size': num_out_feats,
                  'lr_rt': args.learning_rate,
                  'epochs': args.epochs,
                  'trnsfr_model': args.arch,
                  'model_clss': model.classifier,
                  'model_clss_state_dict': model.classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                  }

    #make checkpoint folder if id doenst exist
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    chkpnt_path = args.save_dir + '/chkpnt_{dt:%Y}-{dt:%m}-{dt:%d}_{dt:%H}-{dt:%M}-{dt:%S}.pth'.format(dt=datetime.now())
    torch.save(checkpoint, chkpnt_path)
    print('Checkpoint saved at:\n{}'.format(chkpnt_path))
    
    
if __name__ == '__main__':
    
    #setup command line arguments 
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-sd", "--save_dir", type=str, default="checkpoints", required=False)
    parser.add_argument("-a", "--arch", type=str, default="vgg13", required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, required=False)
    parser.add_argument("-hu", "--hidden_units", type=int, default=512, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=5, required=False)
    parser.add_argument("-g", "--gpu", required=False, action="store_true")
    parser.add_argument('args', nargs='+')
    args = parser.parse_args(args)
    
    #get command line values
    data_dir = args.args[0]
    save_dir = args.save_dir
    arch = args.arch
    lr_rt = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu_requested = args.gpu
    
    #get data loaders
    trainloader, validloader, testloader = get_data_loaders(data_dir)
    
    if gpu_requested and not torch.cuda.is_available():
        #if user requested to use GPU and it is not avilable, then raise exception
        raise Exception("GPU requested but not available. Either set --gpu to False or enable a GPU in your enviornment.")
    else:    
        #else, set device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() and gpu_requested else "cpu")
    
    print("\nUsing {} as device.\n".format(device))
    if device.type == 'cuda':
        print('GPU Name: {}'.format(torch.cuda.get_device_name(0))) #GPU name
        print('(major,minor): {}'.format(torch.cuda.get_device_capability(0))) #major,minor version
        print('GB vRAM: {}\n'.format(torch.cuda.get_device_properties(0).total_memory/1073741824)) #GB of vRAM
    
    #setup model using pretrained arch. Modify classifier with hidden units. Send to device.
    model, num_in_feats, num_out_feats = model_setup(arch, hidden_units, device, data_dir)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_rt) # Only train the classifier parameters, feature parameters are frozen
    
    model, optimizer = model_train(model, epochs, criterion, optimizer, trainloader, validloader, device)
    
    save_model(model, num_in_feats, num_out_feats, args, optimizer)
    print("\nFinished")
    
    
    