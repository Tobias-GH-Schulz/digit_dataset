
import cv2
import imutils
from imutils.perspective import four_point_transform
from imutils import grab_contours
import numpy as np
import torch
from skimage.segmentation import clear_border
from torchvision import transforms
from PIL import Image
#to delete only testing porpuses
import os
os.environ['DISPLAY'] = ':0'

class sudoku_extractor():
    def __init__(self, image_path):
        self.sudoku = cv2.imread(image_path)
        
    def get_array_cells(self, export = False, path = ''):
        original_sudoku, sudoku_cropped = self.find_puzzle(self.sudoku)
        stepX = sudoku_cropped.shape[1] // 9
        stepY = sudoku_cropped.shape[0] // 9
        cell_locations = []
        arr_of_cells_images = []
        idx = 0

        transform_vec = transforms.Compose([
                        transforms.Resize((28,28)),
                            #transforms.Grayscale(),
                            transforms.ToTensor()
        ])
        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                idx += 1
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                leftX = x * stepX
                topY = y * stepY
                rightX = (x + 1) * stepX
                bottomY = (y + 1) * stepY
                # add the (x, y)-coordinates to our cell locations list
                row.append((leftX, topY, rightX, bottomY))
                
                cell = sudoku_cropped[topY:bottomY, leftX:rightX]
                digit = self.detect_digit(cell)

                if digit is not None:
                    #digit = cv2.dilate(digit, (1,1), iterations=1)
                    sharpening_kernel = np.ones((3, 3), np.float32) * -1
                    sharpening_kernel[1, 1] = 9
                    sharp_img = cv2.filter2D(digit, -1, sharpening_kernel)
                  
                    
                    #set export to true if you want to export image to folder
                    if export is True: cv2.imwrite(path + str(idx) + ".png", sharp_img)
                    temp_torch = transform_vec(Image.fromarray(sharp_img))
                    arr_of_cells_images.append(temp_torch)
                
                else:
                    empty_img = np.zeros((28,28))
                    temp_torch = transform_vec(Image.fromarray(empty_img))
                    arr_of_cells_images.append(temp_torch)
            
            cell_locations.append(row)
        ret_tensor = torch.stack(arr_of_cells_images)
        
        return ret_tensor

    def find_puzzle(self, image):
        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # apply adaptive thresholding and then invert the threshold map

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        # find contours in the thresholded image and sort them by size in
    # descending order
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # initialize a contour that corresponds to the puzzle outline
        puzzleContour = None
    # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if the approximated contour has four points, then we can
            # assume we have found the outline of the puzzle
            if len(approx) == 4:
                puzzleContour = approx
                break

        # if the puzzle contour is empty then the script could not find
        # the outline of the Sudoku puzzle so raise an error
        if puzzleContour is None:
            raise Exception(("Could not find Sudoku puzzle outline."))

        # apply a four point perspective transform to both the original
        # image and grayscale image to obtain a top-down bird's eye view
        # of the puzzle
        puzzle = four_point_transform(image, puzzleContour.reshape(4, 2))
        warped = four_point_transform(gray, puzzleContour.reshape(4, 2))

        return puzzle, warped

    def get_contour_x_center_coordinate(contours):
        if cv2.contourArea(contours) > 2:
            M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))

    def detect_digit(self, cell):
        th_cell = cv2.threshold(
            cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        th_cell = clear_border(th_cell)

        # find contours in the thresholded cell
        contours = cv2.findContours(
            th_cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # if no contours were found than this is an empty cell
        if len(contours) == 0:
            return None

        # otherwise, find the largest contour in the cell and create a
        # mask for the contour
        main_cell_contour = max(contours, key=cv2.contourArea)
        background = np.zeros(th_cell.shape, dtype="uint8")
        cv2.drawContours(background, [main_cell_contour], -1, 255, -1)

        # compute the percentage of masked pixels relative to the total
        # area of the image
        (h, w) = th_cell.shape
        percentFilled = cv2.countNonZero(background) / float(w * h)
        # if less than 3% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        if percentFilled < 0.03:
            return None
        # apply the mask to the thresholded cell
        digit = cv2.bitwise_and(th_cell, th_cell, mask=background)

        return digit

if __name__ == '__main__':
    sudoku_extract = sudoku_extractor(
        'digit_from_grid/bin1.1/not_working_img/01042021_1636.jpg')
    array = sudoku_extract.get_array_cells(export=True, path='digit_from_grid/bin1.1/temp_file/')
    for image in array:
        print(image.shape)
        break
    
    
