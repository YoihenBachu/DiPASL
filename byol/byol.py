import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import glob, os
import kornia
import torch.nn as nn
from tqdm import tqdm
import warnings, argparse

warnings.filterwarnings(action = "ignore")

class Byol_dataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformation = self.transform(image=img0)
            dataset_image = transformation["image"]
        return dataset_image.type(torch.FloatTensor)
    

if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_EPOCHS = 80
    ext = ".jpg"

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

    img_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    augment_fn1 = nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(degrees=10)
    )

    augment_fn2 = nn.Sequential(
        kornia.augmentation.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
    )

    base_iter = os.path.join(base_dir, "**")
    img_paths = glob.glob(os.path.join(base_iter, "*"+ext))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Byol_dataset(img_paths, img_transform)
    sampleloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    resnet = models.resnet18(pretrained=True)
    learner = BYOL(
        resnet,
        image_size = 256,
        hidden_layer = 'avgpool',
        projection_size = 256,           # the projection size
        projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
        augment_fn=augment_fn1,
        augment_fn2=augment_fn2,
        moving_average_decay = 0.99,
        use_momentum = True
    ).to(device)

    # define the optimizer and learning rate scheduler
    base_lr = 0.2 * BATCH_SIZE / 256
    optimizer = torch.optim.SGD(learner.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(sampleloader), T_mult=1, eta_min=0.0001, last_epoch=-1, verbose=False)

    # define the data loader
    dataset = Byol_dataset(img_paths, img_transform)  # create dataset with augment_fn
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    # train the model
    accumulation_batch_size = 64
    accumulation_steps = accumulation_batch_size // BATCH_SIZE
    running_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, images in pbar:
            images = images.to(device)
            loss = learner(images)
            if (i + 1) % accumulation_steps == 0:
                loss = loss / accumulation_steps  # scale the loss to account for accumulation_steps
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learner.update_moving_average()
                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / accumulation_steps})
                running_loss = 0.0  # reset running loss after parameter update
            else:
                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / ((i + 1) % accumulation_steps)})

            # update the learning rate
            lr_scheduler.step()

    # save the trained model
    torch.save(learner.state_dict(), os.path.join(model_savepath, 'byol.pt'))