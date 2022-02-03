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

'''
import tensorflow as tf
from tensorflow import keras
import pytesseract
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import random
import imutils


DATADIR = "./dataset/New"
CATEGORIES = ['LicensePlate', 'Not_LicensePlate']

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                #convert my image to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 13, 15, 15) 
                thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,3)
                contours,h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
                largest_rectangle = [0,0]
                for cnt in contours:
                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
                    if len(approx)==4:
                        area = cv2.contourArea(cnt)
                        if area > largest_rectangle[0]:
                            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
                    
                x,y,w,h = cv2.boundingRect(largest_rectangle[1])
                roi=cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 0, 255), 1)
                #roi=img_array[y:y+h,x:x+w]
                #cv2.drawContours(img,[largest_rectangle[1]],0,(0,0,255),-1)
                plt.imshow(roi, cmap = 'gray')
                plt.show()
                training_data.append([roi,class_num])
            except Exception as e:
                pass
create_training_data()
'''






'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pytesseract
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import random


DATADIR = "./dataset"
CATEGORIES = ['LicensePlate', 'Not_LicensePlate']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) 
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array,cmap="gray")
        #plt.show()
        #print(img_array.shape)
        break
    break


IMG_Size = 100

new_array = cv2.resize(img_array,(IMG_Size,IMG_Size))
#plt.imshow(new_array,cmap="gray")
#plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_Size,IMG_Size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

#print(len(training_data))

# tasowanie listy ( zmienia kolejnosc elementow na liscie )
random.shuffle(training_data)

# Zestaw funkcji
X = []
# Moj zestaw etykiet
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_Size, IMG_Size, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
'''

'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pytesseract
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import random
import imutils


DATADIR = "./dataset/New"
CATEGORIES = ['LicensePlate', 'Not_LicensePlate']

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                #convert my image to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 13, 15, 15) 
                Canny = cv2.Canny(gray, 30, 200)
                thresh = cv2.adaptiveThreshold(Canny,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                contours,h = cv2.findContours(Canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                largest_rectangle = [0,0]
                for cnt in contours:
                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
                    if len(approx)==4:
                        area = cv2.contourArea(cnt)
                        if area > largest_rectangle[0]:
                            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

                x,y,w,h = cv2.boundingRect(largest_rectangle[1])
                roi=cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 0, 255), 1)
                #roi=img_array[y:y+h,x:x+w]
                #cv2.drawContours(img,[largest_rectangle[1]],0,(0,0,255),-1)
                plt.imshow(roi, cmap = 'gray')
                plt.show()
                training_data.append([roi,class_num])
            except Exception as e:
                pass
create_training_data()
'''