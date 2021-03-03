from cv2 import cv2
import numpy as np
import os 
import math

def rotate_image(image, angle):
    """
        Rotates the given image by the input angle in the counter-clockwise direction
        Parameters
        ----------
            image : ndim np.array
                image to be rotated
            angle : float
                angle of rotation as degrees.
        Returns
        -------
            rotated image as np.array
    """
    # create an tuple that contains height/2, width/2
    image_center = tuple(np.array(image.shape[1::-1]) / 2) 
    # rot_mat 2x3 rotation mattrix
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # apply the affine transformation to the image
    # size of the output image image.shape[1::-1]
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

img1_filename = '0023_4.jpg'
img2_filename = '0037_4.jpg'
img3_filename = 'img1.jpg'

img_path = os.path.join(os.getcwd(),img3_filename)
img = cv2.imread(img_path)

# resize the image by quartering it
scale_percent = 18
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
img = cv2.resize(img, dsize)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(gray,50,150,apertureSize=3)
# np.pi/180 = 0.0174
# threshold=100 is the vote threshold count
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)

# cv2.imshow('thresh', thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 2))
dilate = cv2.dilate(thresh, kernel, iterations=2)
# cv2.imshow('dilate', dilate)
lines = cv2.HoughLinesP(dilate,rho=1,theta=np.pi/180,threshold=30,minLineLength=5,maxLineGap=20)
img_copy = np.copy(img)
# print(lines)
red = (255,0,0)
green = (0,255,0)
count =0
nb_lines = len(lines)
angle = 0

for line in lines:
    x1,y1,x2,y2 = line[0]
    angle += math.atan2((y2-y1),(x2-x2)) 
    if(count%2):
        cv2.line(img_copy,(x1,y1),(x2,y2),green,2)
    else:
        cv2.line(img_copy,(x1,y1),(x2,y2),red,2)

    count = count + 1
    # cv2.rectangle(img_copy, (x1,y1),(x2,y2),(255,255,0),2)
angle/=nb_lines
print(angle)

rotated = rotate_image(img, angle-1)

cv2.imshow('raw img',img)
cv2.imshow('gray',gray)
cv2.imshow('thresh',thresh),
cv2.imshow('dilate',dilate),
cv2.imshow('rotated',rotated)
cv2.imshow('detected lines',img_copy)


cv2.waitKey(0)
cv2.destroyAllWindows()
