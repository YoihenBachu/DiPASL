import torch.nn as nn
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os, glob
import wandb
import warnings

from utils import *
from dataset import ASL_dataset
from loopers import looper
import config

warnings.filterwarnings(action = "ignore")
        
if __name__ == "__main__":
    base_dir = r'F:\fyp\dataset'
    model_savepath = r'F:\fyp'
    
    if config.WANDB_LOG == True:
        wandb.init(project = config.WANDB_INIT)

    backbones = config.BACKBONE
    epochs = config.EPOCHS
    lr = config.LEARNING_RATE
    opt_method = config.OPTIMIZER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    ext = config.EXTENSION
    iterable_path = os.path.join(base_dir, "**")
    data_paths = glob.glob(os.path.join(iterable_path, "*" + ext))

    df = make_df(data_paths, config.ALPHABETS)
    img_transform = transforms.Compose([transforms.ToTensor()])

    train_df, test_df = train_test_split(df, test_size = 0.3, shuffle = True)
    trainset = ASL_dataset(train_df, img_transform, config.ALPHABETS)
    testset = ASL_dataset(test_df, img_transform, config.ALPHABETS)
    trainloader = DataLoader(trainset, batch_size = config.BATCH_SIZE, drop_last = True)
    testloader = DataLoader(testset, batch_size = config.BATCH_SIZE)

    model = initialize_model(backbones, device, transfer = config.TRANSFER)
    optimizer = initialize_optimizer(opt_method, model, lr)
    loss_fn = nn.CrossEntropyLoss()

    bname = str(backbones)
    wandb_log = config.WANDB_LOG
    looper(epochs,
           optimizer,
           model,
           trainloader,
           testloader,
           loss_fn,
           device,
           wandb_log,
           model_savepath,
           bname,
           opt_method,
           lr)