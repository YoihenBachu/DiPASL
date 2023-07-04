import torch.nn as nn
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os, glob
import wandb
import warnings
import argparse

from utils import *
from dataset import ASL_dataset
from loopers import looper
import config

warnings.filterwarnings(action = "ignore")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        default=config.DATASET_PATH,
        required=False,
        help="folder directory of the dataset, it should follow the directory structure and should contain NUM_CLASS subfolders"
    )

    parser.add_argument(
        "--model_saving_path",
        "-m",
        type=str,
        default=config.MODEL_SAVEPATH,
        required=False,
        help="path of the folder where trained models are to be saved"
    )

    args = parser.parse_args()
    base_dir = args.dataset_dir
    model_savepath = args.model_saving_path
    
    if config.DATASET_TYPE == "skeleton":
        MEAN = [0.5172, 0.4853, 0.4789]
        STD = [0.2236, 0.2257, 0.2162]
    else:
        MEAN = [0.5016, 0.4767, 0.4698]
        STD = [0.2130, 0.2169, 0.2069]
    
    if config.WANDB_LOG == True:
        wandb.init(project = config.WANDB_INIT)

    backbone_name = config.TRAIN_BACKBONE
    backbone = generate_backbone_name(backbone_name)
    epochs = config.EPOCHS
    lr = config.LEARNING_RATE
    opt_method = config.OPTIMIZER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    ext = config.EXTENSION
    iterable_path = os.path.join(base_dir, "**")
    data_paths = glob.glob(os.path.join(iterable_path, "*" + ext))

    df = make_df(data_paths)
    img_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_df, test_df = train_test_split(df, test_size = 0.3, shuffle = True)
    trainset = ASL_dataset(train_df, img_transform)
    testset = ASL_dataset(test_df, img_transform)
    trainloader = DataLoader(trainset, batch_size = config.BATCH_SIZE, drop_last = True)
    testloader = DataLoader(testset, batch_size = config.BATCH_SIZE)

    model = initialize_model(backbone, device, transfer = config.TRANSFER)
    optimizer = initialize_optimizer(opt_method, model, lr)
    loss_fn = nn.CrossEntropyLoss()

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
           backbone,
           opt_method,
           lr)