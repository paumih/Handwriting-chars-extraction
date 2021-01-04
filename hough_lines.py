from cv2 import cv2
import numpy as np
import os 
img1_filename = '0023_4.jpg'
img2_filename = '0037_4.jpg'
img3_filename = '0019_3.jpg'

img_path = os.path.join(os.getcwd(),img3_filename)
img = cv2.imread(img_path)

# resize the image by quartering it
scale_percent = 18
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
img = cv2.resize(img, dsize)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize=3)
# np.pi/180 = 0.0174
# threshold=100 is the vote threshold count
lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=25,minLineLength=5,maxLineGap=120)
img_copy = np.copy(img)
# print(lines)
red = (255,0,0)
green = (0,255,0)
count =0
for line in lines:
    x1,y1,x2,y2 = line[0]
    if(count%2):
        cv2.line(img_copy,(x1,y1),(x2,y2),green,2)
    else:
        cv2.line(img_copy,(x1,y1),(x2,y2),red,2)

    count = count + 1
    # cv2.rectangle(img_copy, (x1,y1),(x2,y2),(255,255,0),2)
 
cv2.imshow('raw img',img)
cv2.imshow('gray',gray)
cv2.imshow('edges',edges)
cv2.imshow('detected lines',img_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()
