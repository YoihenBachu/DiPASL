from torch.utils.data import Dataset
import cv2
from PIL import Image

from utils import onehotencoder
import config
import warnings

warnings.filterwarnings(action = "ignore")

class ASL_dataset(Dataset):
    def __init__(self, df_data, transforms):
        self.data = df_data
        self.img_transform = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        letter = self.data.iloc[idx, 1]
        index = config.ALPHABETS.index(letter)
        label = onehotencoder(index)
        if self.img_transform:
            image = self.img_transform(image)
        return image, label