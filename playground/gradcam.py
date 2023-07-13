import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
import os, glob, shutil
import sys
from PIL import Image

sys.path.append(r'D:\\final_year_project\\DiPASL')
from utils import *
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        required=True,
        help="path of the weight that is supposed to be loaded for prediction",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        required=True,
        help="backbone of the model architecture that is to be used for prediction",
    )
    
    parser.add_argument(
        "--dataset_type",
        "-d",
        type=str,
        default=config.DATASET_TYPE,
        required=False,
        help="backbone of the model architecture that is to be used for prediction",
    )

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    dataset_type = args.dataset_type
    
    img_paths = glob.glob(os.path.join(input_folder, 'images', '*.png'))
    weight_paths = glob.glob(os.path.join(input_folder, 'weights', '*.pt'))
    
    if dataset_type == 'skeleton':
        MEAN = [0.5172, 0.4853, 0.4789]
        STD = [0.2236, 0.2257, 0.2162]
    else:
        MEAN = [0.5016, 0.4767, 0.4698]
        STD = [0.2130, 0.2169, 0.2069]
        
    img_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean = MEAN,
                            std = STD),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for path in img_paths:
        fname = os.path.basename(path).split('.')[0]
        label_index = config.ALPHABETS.index(fname)
        saveroot = os.path.join(output_folder, fname)
        if os.path.isdir(saveroot):
            shutil.rmtree(saveroot)
        os.makedirs(saveroot)
        
        img0 = cv2.imread(path)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(np.uint8(img))
        img_tensor = img_transform(img_pil)
        img_inp = img_tensor.unsqueeze(0).to(device)
        
        for weight in weight_paths:
            backbone_name = os.path.basename(weight).split('_')[0]
            backbone = generate_backbone_name(backbone_name)
            model = load_model(backbone, weight, device)
            
            target_layer = generate_gradcam_layer(model, backbone)
            # print(target_layer)
            cam = GradCAM(model=model, target_layers=target_layer)
            targets = [ClassifierOutputTarget(label_index)]
            grayscale_cam = cam(input_tensor=img_inp, targets=targets, aug_smooth=True, eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            
            grayscale_cam_uint8 = (grayscale_cam * 255).astype('uint8')
            colored_cam = cv2.applyColorMap(grayscale_cam_uint8, cv2.COLORMAP_JET)
            output_img = cv2.addWeighted(img, 0.7, colored_cam, 0.5, 0)

            filename = str(fname) + '_' + str(os.path.basename(weight).replace('.pt', '.png'))
            cv2.imwrite(os.path.join(saveroot, filename), output_img)