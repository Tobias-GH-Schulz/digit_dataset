import os
import random
from scipy import ndarray

# image processing library
import cv2
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import matplotlib


fonts = ["Arial", "Calibri", "Verdana", 
        "Times New Roman", "Rockwell", "Helvetica",
        "Garamond", "Futura", "Franklin Gothic", "Cambria"]

for font in fonts:
    for number in range(0,10):
        folder_path_input = f"./printed_digits/train_images/{font}/{number}"
        folder_path_output = f"./printed_digits/all_train_images/{number}"

        # find all files paths from the folder
        images = [os.path.join(folder_path_input, f) for f in os.listdir(folder_path_input) if os.path.isfile(os.path.join(folder_path_input, f))]
        images = [ x for x in images if ".DS_Store" not in x ]

        idx = 0
        for i in images:
            idx += 1
            image_to_copy = io.imread(i, plugin="matplotlib")

            new_file_path = f'%s/{font}_{number}_%s.jpg' % (folder_path_output, idx)
            io.imsave(new_file_path, image_to_copy)