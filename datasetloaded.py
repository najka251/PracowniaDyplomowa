import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import cv2
import os
import imutils
import numpy as np

DATADIR = "./dataset/New/LicensePlate/"

training_data = []

def create_training_data():
    path = os.path.join(DATADIR)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            #convert my image to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray,11, 17, 17)
            thresh = cv2.Canny(gray,30,200)
            #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV)
            points = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(points)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            roi = None
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 10, True)
                if len(approx) == 4 :
                    roi = approx
                    break
            #Black color on everything, but not on contours which we choose 
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [roi], 0,255, -1)
            new_image = cv2.bitwise_and(img_array, img_array, mask=mask)
            
            # Crop license plate
            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            img_crop = gray[x1:x2+1, y1:y2+1]
            
            #Read Text - Easy OCR
            
            reader = easyocr.Reader(['en'])
            #en because we don't PL letters on license plate
            result = reader.readtext(img_crop)
            
            #Render Result
            text_from_crop = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            #Add license plate numbers on picture
            finish = cv2.putText(img_array, text=text_from_crop, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            #Add a rectangle around the license plate
            finish = cv2.rectangle(img_array, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        
            #Show result
            plt.imshow(cv2.cvtColor(finish, cv2.COLOR_BGR2RGB))
            plt.show()
        except Exception as e:
            pass
create_training_data()

