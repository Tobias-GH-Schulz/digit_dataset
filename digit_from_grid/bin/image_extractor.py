import cv2
import imutils
from imutils import contours as cts
import numpy as np
import os
os.environ['DISPLAY'] = ':0'

def print_image(image):
    print('here')
    cv2.imshow('thresh', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, contourn, path_string):
    x, y, w, h = cv2.boundingRect(contourn) 
    cropped_contour = img[y: y + h , x: x + w]
    ret,thresh = cv2.threshold(cropped_contour,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2))
    dilated_img = cv2.dilate(thresh,kernel) 
    output = cv2.resize(dilated_img, (28,28), fx=100, fy=100, interpolation=cv2.INTER_CUBIC)
    i = 5 #crop factor
    temp = output[i : 28 - i, i: 28 - i]
    #output = cv2.threshold(temp,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(path_string, temp)
    #print_image(cropped_contour)
    return 

image = cv2.imread('digit_from_grid/grid2.png', 0)
#image = cv2.resize(image,(225,225))
#gray = cv2.cvtColor(image, cv2.COLOR_BG2GRAY)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
print(thresh.shape)
contourns, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contourn in contourns:
    area = cv2.contourArea(contourn)
    if area < 550:
        cv2.drawContours(thresh, [contourn], -1, (0,0,0), -1)


vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=5)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=5) 


invert = 255 - thresh
print_image(invert)
contourns, h = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#print("Here", len(contourns))
#print("Here", invert)
contourns, _ = cts.sort_contours(contourns, method = 'top-to-bottom')
temp_row = []
sudoku_rows = []
print('here')
for (i, c) in enumerate(contourns, 1):
    area = cv2.contourArea(c)
    if area < 4500:
        temp_row.append(c)
        if i % 9 == 0:
            contourns, _ = cts.sort_contours(temp_row, method = "left-to-right")
            sudoku_rows.append(contourns)
            temp_row = []
font_list = ['Helvetica','Calibri','Futura','Garamond','TimesNewRoman','Arial','Cambria','Verdana','Rockwell']
string_path = 'digit_from_grid/img_t/NUMBER_{}_{}.jpg'
for row, i in zip(sudoku_rows, font_list):
    for c, j in zip(row, range(1,10,1)):
        print('here')
        save_image(image.copy(), c, string_path.format(i,j))
        '''
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255,255,255), 3)
        result = cv2.bitwise_and(image, mask)
        result[mask == 0] = 255
        cv2.imshow('result', result)
        cv2.waitKey(0)
        '''
#print_image(invert)
cv2.destroyAllWindows()