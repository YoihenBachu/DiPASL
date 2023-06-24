import torch
from torchvision import transforms

from PIL import Image
import cv2
import warnings
import argparse

import pyttsx3
import config
from utils import load_model
from cvzone.HandTrackingModule import HandDetector

warnings.filterwarnings(action = "ignore")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        "-w",
        type=str,
        required=True,
        help="path of the weight that is supposed to be loaded for prediction",
    )

    parser.add_argument(
        "--backbone",
        "-b",
        type=str,
        required=True,
        help="backbone of the model architecture that is to be used for prediction",
    )

    args = parser.parse_args()
    weight_path = args.weight_path
    backbone = args.backbone

    img_transform = transforms.Compose([transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    imgSize = config.IMG_SIZE
    labels = config.ALPHABETS
    detector = HandDetector(maxHands=1)
    string = ''
    former_letter = ''
    offset = 20

    model = load_model(backbone, weight_path)
    model.eval()
    while True:
        success, img0 = cap.read()
        imgOutput = img0.copy()
        hands, img1 = detector.findHands(img0)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            success, img2 = cap.read()
            imgCrop_normal = img2[y - round(w/2):y + round(w/2) + h, x - round(h/2):x + w + round(h/2)]
            try:
                imgResize_normal = cv2.resize(imgCrop_normal, (imgSize, imgSize))
                cv2.imshow("ImageCrop_normal", imgCrop_normal)
            except:
                pass
            
            img_np = Image.fromarray(imgResize_normal)
            img_tensor = img_transform(img_np)
            image = img_tensor.unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            index = predicted.item()

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                        (x + w + offset, y + h + offset), (255, 0, 255), 4)
            
            key = cv2.waitKey(1)        
            if str(labels[index]) == 'Y':
                print('letter recorded')
                string = string + str(former_letter)

            if str(labels[index]) == 'O':
                print(string)
                pyttsx3.speak(string)
                string = ''

            former_letter = str(labels[index])

        cv2.imshow("Image", imgOutput)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            cap.release()
            break
    
    cv2.destroyAllWindows()
    
        