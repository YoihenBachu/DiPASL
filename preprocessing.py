import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import os

import config
 
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
 
imgSize = config.IMG_SIZE

folder1 = os.path.join(config.FOLDER_ROOT1, config.BASE_LETTER)
folder2 = os.path.join(config.FOLDER_ROOT2, config.BASE_LETTER)

counter = 0
 
while True:
    success, img0 = cap.read()
    hands, img1 = detector.findHands(img0)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop_line = img1[y - round(w/2):y + round(w/2) + h, x - round(h/2):x + w + round(h/2)]
        try:
            imgResize_line = cv2.resize(imgCrop_line, (imgSize, imgSize))
            cv2.imshow("ImageCrop_line", imgCrop_line)
        except:
            pass

        success, img2 = cap.read()
        imgCrop_normal = img2[y - round(w/2):y + round(w/2) + h, x - round(h/2):x + w + round(h/2)]
        try:
            imgResize_normal = cv2.resize(imgCrop_normal, (imgSize, imgSize))
            cv2.imshow("ImageCrop_normal", imgCrop_normal)
        except:
            pass


    cv2.imshow("Image", img1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        counter += 1
        time_param = str(time.time())
        cv2.imwrite(os.path.join(folder1, 'Image_' + time_param + config.EXTENSION), imgResize_line)
        cv2.imwrite(os.path.join(folder2, 'Image_' + time_param + config.EXTENSION), imgResize_normal)
        print(f"Count of images for the letter {config.BASE_LETTER} : {counter}")
    elif key == 27:
        cap.release()
        break

cv2.destroyAllWindows()