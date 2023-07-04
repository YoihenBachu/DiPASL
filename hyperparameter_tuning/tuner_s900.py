from tqdm import tqdm
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms
import os, glob
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
import pandas as pd
import cv2
from PIL import Image

NUM_CLASS = 26
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0001
alphabets = ['A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X',
            'Y', 'Z']

sweep_configuration = {
    "method": "bayes",
    "name": "hyperparameters search",
    "metric": {'goal': 'maximize', 'name': 'Training accuracy'},
    "parameters": {
        "epochs": {"values": [30]},
        "lr": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "backbones": {"values": ["resnet18", "mobilenetv2_050", "efficientnetv2_rw_m", "rexnet_100", "xception41"]},
        "optims": {"values": ["adam", "sgd", "adamw"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sign Language Sweep")

def initialize_optimizer(opt_method, model, lr):
    if opt_method == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if opt_method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if opt_method == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, nesterov=True, momentum = 0.9
        )
    
def initialize_model(backbone, device):
    if backbone == 'resnet18':
        original_model = timm.create_model('resnet18', pretrained=True)
        num_features = original_model.fc.in_features
        original_model.fc = nn.Linear(num_features, NUM_CLASS)
        model = original_model.to(device)
        
    elif backbone == 'xception41':
        original_model = timm.create_model('xception41', pretrained=True)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, NUM_CLASS)
        model = original_model.to(device)
        
    elif backbone == 'rexnet_100':
        original_model = timm.create_model(backbone, pretrained=True)
        original_model.global_pool = nn.AdaptiveAvgPool2d(1)
        num_features = original_model.head.fc.in_features
        original_model.head.fc = nn.Linear(num_features, NUM_CLASS)
        model = original_model.to(device)
        
    else:
        original_model = timm.create_model(backbone, pretrained=True)
        num_features = original_model.classifier.in_features
        original_model.classifier = nn.Linear(num_features, NUM_CLASS)
        model = original_model.to(device)
    return model
    
def onehotencoder(value):
    one_hot_vector = np.zeros(26)
    one_hot_vector[value] = 1
    one_hot_vector = one_hot_vector.astype(int)

    return one_hot_vector

def make_df(all_path, alphabets):
    c = 0
    df = pd.DataFrame(columns = ['img_path', 'label'])
    for data_path in all_path:
        base = os.path.split(data_path)[0]
        label = os.path.basename(base)
        if label in alphabets:
            df.loc[c] = [data_path, label]
            c+=1
    return df

class ASL_dataset(Dataset):
    def __init__(self, df_data, transforms, alphabets):
        self.data = df_data
        self.img_transform = transforms
        self.alphabets = alphabets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        letter = self.data.iloc[idx, 1]
        index = self.alphabets.index(letter)
        label = onehotencoder(index)
        if self.img_transform:
            image = self.img_transform(image)
        return image, label
    

def looper(epochs,
           optimizer,
           model,
           trainloader,
           testloader,
           loss_fn,
           device,
           wandb_log,
           model_savepath,
           bname,
           optim,
           lr
):
    e = 0
    for epoch in range(epochs):
        e += 1
        print('Epoch {}/{}, lr:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        print('-' * 30)
        # Train the model
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        model.train()  # Set the model to training mode
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for (_, data) in pbar:
            pbar.set_description(f"Epoch {epoch+1} | Learning Rate: {optimizer.param_groups[0]['lr']}")
            images = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            out = model(images)
            truth = torch.max(labels, dim=1)[1]
            loss = loss_fn(out, truth)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(truth.cpu().numpy())

            pbar.set_postfix({'TrainLoss': loss.item()})
            if wandb_log:
                wandb.log({"Training loss": loss.item()})

        train_acc = accuracy_score(train_labels, train_predictions)
        train_loss = train_loss / len(trainloader)

        # Evaluate the model
        test_loss = 0.0
        test_predictions = []
        test_labels = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pbar = tqdm(enumerate(testloader), total=len(testloader))
            for (_, data) in pbar:
                pbar.set_description(f"Epoch {epoch+1} | Learning Rate: {optimizer.param_groups[0]['lr']}")
                images = data[0].to(device)
                labels = data[1].to(device)

                out = model(images)
                truth = torch.max(labels, 1)[1]
                loss = loss_fn(out, truth)

                test_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(truth.cpu().numpy())

                pbar.set_postfix({'TestLoss': loss.item()})
                if wandb_log:
                    wandb.log({"Testing loss": loss.item()})
                    wandb.log({"Training accuracy": train_acc})

        test_acc = accuracy_score(test_labels, test_predictions)
        test_loss = test_loss / len(testloader)

        filename = str(bname) + '_' + str(optim) + '_' + str(lr) + '_' + str(epoch) + '.pt'
        if ((e) >= 20) and ((e)%5 == 0):
            torch.save(model.state_dict(), os.path.join(model_savepath, str(filename)))
            wandb.alert(
                title = 'Update',
                text = f'Epoch: {epoch+1}\nTraining Loss: {train_loss} \nValidation Loss: {test_loss} \nAccuracy: {train_acc} \nModel saved at {filename}',
            )
        else:
            wandb.alert(
                title = 'Update',
                text = f'Epoch: {epoch+1}\nTraining Loss: {train_loss} \nValidation Loss: {test_loss} \nAccuracy: {train_acc}',
        )
                
        # Print the training and testing statistics
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%'
            .format(epoch+1, epochs, train_loss, train_acc*100, test_loss, test_acc*100))
        
def main():
    base_dir = r'F:\fyp\dataset'
    model_savepath = r'F:\fyp'
    
    wandb.init()

    backbones = wandb.config.backbones
    epochs = wandb.config.epochs
    lr = wandb.config.lr
    opt_method = wandb.config.optims

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    ext = ".png"
    iterable_path = os.path.join(base_dir, "**")
    data_paths = glob.glob(os.path.join(iterable_path, "*" + ext))

    df = make_df(data_paths, alphabets)
    img_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5172, 0.4853, 0.4789],
                            std = [0.2236, 0.2257, 0.2162]),
    ])
    
    train_df, test_df = train_test_split(df, test_size = 0.3, shuffle = True)
    trainset = ASL_dataset(train_df, img_transform, alphabets)
    testset = ASL_dataset(test_df, img_transform, alphabets)
    trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, drop_last = True)
    testloader = DataLoader(testset, batch_size = BATCH_SIZE)

    model = initialize_model(backbones, device)
    optimizer = initialize_optimizer(opt_method, model, lr)
    loss_fn = nn.CrossEntropyLoss()

    bname = str(backbones)
    wandb_log = True
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
    
    torch.cuda.empty_cache()

wandb.agent(sweep_id, main, count=60)
wandb.finish()