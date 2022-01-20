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
                img_array2 = img_array.copy()
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 7, 75, 75)
                thresh = cv2.Canny(gray, 70, 400)
                #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV)
                contours,h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
                _= cv2.drawContours(img_array2,contours, -1, (255, 0, 0), 2)
                largest_rectangle = [0,0]
                roi = None
                for cnt in contours:
                    peri = cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt, 0.02* peri, True)
                    if len(approx)==4 :
                        area = cv2.contourArea(cnt)
                        if area > largest_rectangle[0]:
                            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
                    
                x,y,w,h = cv2.boundingRect(largest_rectangle[1])
                roi=cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 255), 2)
                #roi=img_array[y:y+h,x:x+w]
                #cv2.drawContours(img,[largest_rectangle[1]],0,(0,0,255),-1)
                plt.imshow(img_array2, cmap = 'gray')
                plt.show()
                training_data.append([roi,class_num])
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