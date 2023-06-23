import numpy as np
import pandas as pd
import os
import warnings
import torch
import timm
import torch.nn as nn

import config
warnings.filterwarnings(action = "ignore")

def onehotencoder(value):
    one_hot_vector = np.zeros(config.NUM_CLASS)
    one_hot_vector[value] = 1
    one_hot_vector = one_hot_vector.astype(int)

    return one_hot_vector

def make_df(all_path):
    c = 0
    alphabets = config.ALPHABETS
    df = pd.DataFrame(columns = ['img_path', 'label'])
    for data_path in all_path:
        base = os.path.split(data_path)[0]
        label = os.path.basename(base)
        if label in alphabets:
            df.loc[c] = [data_path, label]
            c+=1
    return df

def initialize_optimizer(opt_method, model, lr):
    if opt_method == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    if opt_method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    if opt_method == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY, nesterov=True, momentum = 0.9
        )
    
def initialize_model(backbone, device, transfer):
    if backbone == 'resnet18':
        if transfer == True:
            original_model = timm.create_model(backbone, pretrained=True)
        else:
            original_model = timm.create_model(backbone, pretrained=False)
        num_ftrs = original_model.fc.in_features
        original_model.fc = nn.Linear(num_ftrs, config.NUM_CLASS)
        model = original_model.to(device)
    elif backbone == 'xception41':
        if transfer == True:
            original_model = timm.create_model(backbone, pretrained=True)
        else:
            original_model = timm.create_model(backbone, pretrained=False)
        original_model.add_module('fc', nn.Linear(2048, config.NUM_CLASS))
        model = original_model.to(device)
    else:
        if transfer == True:
            original_model = timm.create_model(backbone, pretrained=True)
        else:
            original_model = timm.create_model(backbone, pretrained=False)
        original_model.add_module('fc', nn.Linear(1000, config.NUM_CLASS))
        model = original_model.to(device)
    return model

def load_model(backbone, weight):
    if backbone == 'resnet18':
        model = timm.create_model(backbone, pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.NUM_CLASS)
        model.load_state_dict(torch.load(weight))

    elif backbone == 'xception41':
        model = timm.create_model(backbone, pretrained=True)
        model.add_module('fc', nn.Linear(2048, config.NUM_CLASS))
        model.load_state_dict(torch.load(weight))

    else:
        model = timm.create_model(backbone, pretrained=True)
        model.add_module('fc', nn.Linear(1000, config.NUM_CLASS))
        model.load_state_dict(torch.load(weight))

    return model