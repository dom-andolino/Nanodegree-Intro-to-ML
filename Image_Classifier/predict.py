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


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    img_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = img_transforms(Image.open(image_path))
    
    return(image)


def load_checkpoint(chkpnt_path, device):
    '''
    load model from checkpoint file
    send to device
    return model
    '''
    
    checkpoint = torch.load(chkpnt_path)

    #use transfer learning model as start
    model = eval("models."+checkpoint['trnsfr_model']+"(pretrained=True)")

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['model_clss']
    model.classifier.load_state_dict(checkpoint['model_clss_state_dict'])

    lr_rt = checkpoint['lr_rt']
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_rt)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    model.to(device)
    model.train()
    
    return(model)


def predict(image, model, top_k, device, category_names_path):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)

    input_ = image.unsqueeze(0)
    input_ = input_.to(device)

    model = model.eval()

    logps = model(input_)
    ps = torch.exp(logps)
    ps_top = ps.topk(top_k)

    probs = ps_top[0].tolist()[0]
    classes_id = ps_top[1].tolist()[0]
    classes = [cat_to_name[str(k)] for k in classes_id]
    
    return(probs, classes)


if __name__ == '__main__':
    
    #setup command line arguments 
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-tk", "--top_k", type=int, default=3, required=False)
    parser.add_argument("-cn", "--category_names", type=str, default="cat_to_name.json", required=False)
    parser.add_argument("-g", "--gpu", required=False, action="store_true")
    parser.add_argument('args', nargs='+')
    args = parser.parse_args(args)
    
    #get command line values
    image_path = args.args[0]
    chkpnt_path = args.args[1]
    top_k = args.top_k
    category_names_path = args.category_names
    gpu_requested = args.gpu
    
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
        
    
    image = process_image(image_path)
    model = load_checkpoint(chkpnt_path, device)
    probs, classes = predict(image, model, top_k, device, category_names_path)
    pred_dict = dict(zip(classes,[str(round(p*100,1))+'%' for p in probs])) 
    print('\nPredictions:')
    print(pred_dict)
    

