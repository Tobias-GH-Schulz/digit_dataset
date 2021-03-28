import cv2
import numpy as np
from DetectDigit import detect_digit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

def ClassifyDigits(sudoku_gray):
    stepX = sudoku_gray.shape[1] // 9
    stepY = sudoku_gray.shape[0] // 9

    board = np.zeros((9, 9), dtype="int")

    cell_locations = []
    idx = 0
    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            idx +=1
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            leftX = x * stepX
            topY = y * stepY
            rightX = (x + 1) * stepX
            bottomY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((leftX, topY, rightX, bottomY))

            cell = sudoku_gray[topY:bottomY, leftX:rightX]

            # find number in cell
            digit = detect_digit(cell)

            if digit is not None:
                # optimize digits for classifier
                sharpening_kernel = np.ones((3,3), np.float32) * -1
                sharpening_kernel[1,1] = 9
                sharp_img = cv2.filter2D(digit, -1, sharpening_kernel)
                digit_sm = cv2.resize(sharp_img, (28, 28))

                # save cells to check
                cv2.imwrite("./sudoku_images/Cells/" + str(idx) + ".png", digit_sm)

                # reshape 
                digit_reshaped = digit_sm.reshape(1,28,28,1)
                test_digit = (digit_reshaped[...,::-1].astype(np.float32)) / 255.0
                #
                #
                #
                # here we need to put our classifier!!!                 
                # load trained keras model
                model = load_model('./model/')
                # classify the number 
                preds = model.predict(test_digit)
                label = np.argmax(preds, axis=1) 
                board[y, x] = label

        cell_locations.append(row)

    return cell_locations, board