import os
from cv2 import cv2
import numpy as np
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

def read_image(img_path):
    image = cv2.imread(img_path)

    scale_percent = 18  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    rescaled_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) 
    return image, rescaled_img  

def save_img(dir_path,filename,img):
    """
        dir_path - directory path where the image will be saved
        filename - requires a valid image format
        img - image to be saved
    """
    file_path = os.path.join(dir_path,filename)
    cv2.imwrite(file_path,img)

def find_text_angle(dilated_img,org_img):
    """
        org_img - original image
        img - dilated img
    """
    lines = cv2.HoughLinesP(dilated_img,rho=1,theta=np.pi/180,threshold=30,minLineLength=5,maxLineGap=20)

    nb_lines = len(lines)
    angle = 0

    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle += math.atan2((y2-y1),(x2-x1)) 

    angle/=nb_lines

    rotated = rotate_image(org_img, angle-1)
    rot_dilated = rotate_image(dilated_img,angle-1)

    return rotated, rot_dilated
 
def extract_text_lines(img,output_dir):   
    """
        img - image from which the text lines are extracted
        output_dir - directory where the extracted lines should be saved 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    rotated,rot_dilated = find_text_angle(dilate,img)

    cnts = cv2.findContours(rot_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    else:
        cnts = cnts[1]

    lines_path = os.path.join(output_dir,'lines')  

    if not os.path.exists(lines_path):
        os.makedirs(lines_path)

    for line_idx, line in enumerate(cnts, start=-len(cnts)):
        x, y, w, h = cv2.boundingRect(line)
        roi = rotated[y:y + h, x:x + w]
        filename = 'line'+str(line_idx)+ '.jpg'
        save_img(lines_path,filename=filename,img=roi)
    
def extract_text_chars(img,output_dir):
    """
        img - image from which the individual chars are extracted
        output_dir - directory where the extracted lines should be saved 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 11)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))

    dilate = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    else:
        cnts = cnts[1]
    
    chars_path = os.path.join(output_dir,'chars')  

    if not os.path.exists(chars_path):
        os.makedirs(chars_path)
        
    for char_idx, character in enumerate(cnts, start=-len(cnts)):
        x, y, w, h = cv2.boundingRect(character)
        roi = img[y:y + h, x:x + w]
        filename = 'char'+str(char_idx)+ '.jpg'
        save_img(chars_path,filename=filename,img=roi)



if __name__ == '__main__':
    # provide the indut/outpur directory paths
    input_dir = os.path.join(os.getcwd(),'images')
    output_dir = os.path.join(os.getcwd(),'output')  

    for img_file in os.listdir(input_dir):
        img_file_path = os.path.join(input_dir,img_file)
        image, rescaled_image = read_image(img_path=img_file_path)
        img_out_dir = os.path.join(output_dir,img_file.split('.')[0])
        extract_text_lines(rescaled_image,img_out_dir)
        extract_text_chars(image,img_out_dir)