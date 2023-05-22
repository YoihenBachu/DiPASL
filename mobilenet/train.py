import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os, glob
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings, argparse
import datetime, pytz

import config
from utils import *
from dataset import ASL_dataset

warnings.filterwarnings(action = "ignore")
tz_ist = pytz.timezone("Asia/Kolkata")


if __name__ == "__main__":
    e = 0

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        "-b",
        type=str,
        required=True,
        help="input directory that contains all the classes of the sign alphabets",
    )

    parser.add_argument(
        "--model_saving_path",
        "-m",
        type=str,
        required=True,
        help="directory where the model is to be saved",
    )

    args = parser.parse_args()
    base_dir = args.base_dir
    model_savepath = args.model_saving_path

    ext = config.EXTENSION
    iterable_path = os.path.join(base_dir, "**")
    data_paths = glob.glob(os.path.join(iterable_path, "*" + ext))

    print(
        f'{datetime.datetime.now(tz_ist).strftime("%Y-%m-%d %H:%M:%S")}--[INFO]: Started making the dataframe'
    )

    df = make_df(data_paths)
    if config.SAVE_CSV == True:
        df.to_csv(os.path.join(base_dir, "data.csv"), index = False)

    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df = train_test_split(df, test_size = 0.3, shuffle = True)
    trainset = ASL_dataset(train_df, img_transform)
    testset = ASL_dataset(test_df, img_transform)
    trainloader = DataLoader(trainset, batch_size = config.BATCH_SIZE, drop_last = True, shuffle = True)
    testloader = DataLoader(testset, batch_size = config.BATCH_SIZE)

    model = torchvision.models.resnet18(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASS)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum = config.MOMENTUM, nesterov = True)

    print(
        f'{datetime.datetime.now(tz_ist).strftime("%Y-%m-%d %H:%M:%S")}--[INFO]: Model initialised... started training'
    )

    for epoch in range(config.EPOCHS):
        e+=1
        print('Epoch {}/{}, lr:{}'.format(epoch + 1,
            config.EPOCHS, optimizer.param_groups[0]['lr']))
        print('-' * 30)
        # Train the model
        train_loss = 0.0
        correct = 0
        total = 0
        model.train() # Set the model to training mode
        for images, labels in tqdm(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = model(images)
            # preds = torch.max(out, dim = 1)[1].to(dtype = torch.float)
            truth = torch.max(labels, dim = 1)[1]
            loss = loss_fn(out, truth)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
        train_acc = 100 * correct / total
        train_loss = train_loss / len(trainloader)

        # Evaluate the model
        test_loss = 0.0
        correct = 0
        
        total = 0
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                # preds = torch.max(out, dim = 1)[1].to(dtype = torch.float)
                truth = torch.max(labels, 1)[1]
                loss = loss_fn(out, truth)
                
                test_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.max(labels, 1)[1]).sum().item()
        test_acc = 100 * correct / total
        test_loss = test_loss / len(testloader)

        filename = config.MODEL_NAME + str(epoch+1) + '.pt'
        if (e)%5 == 0:
            torch.save(model.state_dict(), os.path.join(model_savepath, str(filename)))
        # Print the training and testing statistics
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%'
            .format(epoch+1, config.EPOCHS, train_loss, train_acc, test_loss, test_acc))
