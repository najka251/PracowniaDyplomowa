import cv2
import numpy as np

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area>500:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                print(len(approx))
                objCor = len(approx)
                x , y , w, h = cv2.boundingRect(approx)
                if objCor == 4:
                    aspRatio =w/float(h)
                    if aspRatio < 0.95 or aspRatio > 1.05: ObjectType="LicensePlate"
                elif objCor>4: ObjectType="Not_LicensePlate"
                else:ObjectType="None"
                cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(imgContour,ObjectType,
                            (x+(w//2)-70,y+(h//2)-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,100,255),2)


path = 'Data/Cars11.png'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
imgCanny = cv2.Canny(imgBlur,30,50)
getContours(imgCanny)

cv2.imshow("Canny", imgCanny)
cv2.imshow("img Contour", imgContour)
cv2.waitKey(0)