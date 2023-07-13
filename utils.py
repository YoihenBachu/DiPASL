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

def generate_backbone_name(input_string):
    backbone_name = input_string.lower()
    if "resnet" in backbone_name:
        return "resnet18"
    elif "xception" in backbone_name:
        return "xception41"
    elif "rexnet" in backbone_name:
        return "rexnet_100"
    elif "efficientnet" in backbone_name:
        return "efficientnetv2_rw_m"
    elif "mobilenet" in backbone_name:
        return "mobilenetv2_050"
    else:
        return "Unidentified backbone name, pick from the following [resnet, mobilenet, efficientnet, xception, rexnet]"

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
            original_model = timm.create_model('resnet18', pretrained=True)
        else:
            original_model = timm.create_model('resnet18', pretrained=False)
        num_features = original_model.fc.in_features
        original_model.fc = nn.Linear(num_features, config.NUM_CLASS)
        model = original_model.to(device)
        
    elif backbone == 'xception41':
        if transfer == True:
            original_model = timm.create_model('xception41', pretrained=True)
        else:
            original_model = timm.create_model('xception41', pretrained=False)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, config.NUM_CLASS)
        model = original_model.to(device)
        
    elif backbone == 'rexnet_100':
        if transfer == True:
            original_model = timm.create_model('rexnet_100', pretrained=True)
        else:
            original_model = timm.create_model('rexnet_100', pretrained=False)
        original_model.global_pool = nn.AdaptiveAvgPool2d(1)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, config.NUM_CLASS)
        model = original_model.to(device)
        
    else:
        if transfer == True:
            original_model = timm.create_model('rexnet_100', pretrained=True)
        else:
            original_model = timm.create_model('rexnet_100', pretrained=False)
        num_features = original_model.classifier.in_features
        original_model.classifier = nn.Linear(num_features, config.NUM_CLASS)
        model = original_model.to(device)
    return model
    
def load_model(backbone, weight, device):
    if backbone == 'resnet18':
        original_model = timm.create_model('resnet18', pretrained=True)
        num_features = original_model.fc.in_features
        original_model.fc = nn.Linear(num_features, config.NUM_CLASS)
        original_model.load_state_dict(torch.load(weight, map_location=device))
        model = original_model.to(device)
        
    elif backbone == 'xception41':
        original_model = timm.create_model('xception41', pretrained=True)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, config.NUM_CLASS)
        original_model.load_state_dict(torch.load(weight, map_location=device))
        model = original_model.to(device)
        
    elif backbone == 'rexnet_100':
        original_model = timm.create_model(backbone, pretrained=True)
        original_model.global_pool = nn.AdaptiveAvgPool2d(1)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, config.NUM_CLASS)
        original_model.load_state_dict(torch.load(weight, map_location=device))
        model = original_model.to(device)
        
    else:
        original_model = timm.create_model(backbone, pretrained=True)
        num_features = original_model.classifier.in_features
        original_model.classifier = nn.Linear(num_features, config.NUM_CLASS)
        original_model.load_state_dict(torch.load(weight, map_location = device))
        model = original_model.to(device)
    return model

def generate_gradcam_layer(model, backbone):
    if backbone == 'rexnet_100':
        layer = [list(model.children())[1][-2].conv_exp.conv,
                 list(model.children())[1][-2].conv_dw.conv,
                 list(model.children())[1][-2].conv_pwl.conv,
                 list(model.children())[1][-1].conv]
    elif backbone == 'resnet18':
        layer = [model.layer4[0].conv1,
                 model.layer4[0].conv2,
                 model.layer4[-1].conv1,
                 model.layer4[-1].conv2]
    elif backbone == 'xception41':
        layer = [model.blocks[-1].stack.conv2.conv_dw,
                 model.blocks[-1].stack.conv2.conv_pw,
                 model.blocks[-1].stack.conv3.conv_dw,
                 model.blocks[-1].stack.conv3.conv_pw]
    else:
        layer = [model.blocks[-1][-1].conv_pw,
                 model.blocks[-1][-1].conv_dw,
                 model.blocks[-1][-1].conv_pwl,
                 model.conv_head]
    return layer