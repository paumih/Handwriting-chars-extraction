import os
from cv2 import cv2

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
    
def extract_text_lines(img,output_dir):   
    """
        img - image from which the text lines are extracted
        output_dir - directory where the extracted lines should be saved 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)

    # cv2.imshow('thresh', thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 2))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    # cv2.imshow('dilate', dilate)
    # cv2.waitKey(0)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    else:
        cnts = cnts[1]

    lines_path = os.path.join(output_dir,'lines')  

    if not os.path.exists(lines_path):
        os.makedirs(lines_path)

    for line_idx, line in enumerate(cnts, start=-len(cnts)):
        x, y, w, h = cv2.boundingRect(line)
        roi = img[y:y + h, x:x + w]
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


    # cv2.imshow('dilate for characters', dilate)
    cv2.waitKey(0)


input_dir = os.path.join(os.getcwd(),'Input Images')
output_dir = os.path.join(os.getcwd(),'Output')

for img_file in os.listdir(input_dir):
    img_file_path = os.path.join(input_dir,img_file)
    image, rescaled_image = read_image(img_path=img_file_path)
    img_out_dir = os.path.join(output_dir,img_file.split('.')[0])
    extract_text_lines(rescaled_image,img_out_dir)
    extract_text_chars(image,img_out_dir)