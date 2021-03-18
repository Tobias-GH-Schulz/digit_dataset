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

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_stretch(image_array: ndarray):
    new_width = random.randint(31, 35)
    # dsize
    dsize = (new_width, image_array.shape[0])

    # resize image
    resized_image = cv2.resize(image_array, dsize, interpolation = cv2.INTER_AREA)

    pad = (new_width - image_array.shape[0]) / 2
    horizontal_stretch = resized_image[int(0):int(resized_image.shape[0]), int(0+pad):int(resized_image.shape[1]-pad)]
    return horizontal_stretch

def vertical_stretch(image_array: ndarray):
    new_width = random.randint(31, 35)
    # dsize
    dsize = (image_array.shape[0], new_width)
    
    # resize image
    resized_image = cv2.resize(image_array, dsize, interpolation = cv2.INTER_AREA)
    
    pad = (new_width - image_array.shape[1]) / 2
    vertical_stretch = resized_image[int(0+pad):int(resized_image.shape[0]-pad), int(0):int(resized_image.shape[1])]
    return vertical_stretch

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_stretch': horizontal_stretch,
    'vertical_stretch': vertical_stretch
}

fonts = ["Arial", "Calibri", "Verdana", 
        "Times New Roman", "Rockwell", "Helvetica",
        "Garamond", "Futura", "Franklin Gothic", "Cambria"]

num_files_desired = int(input("How many pictures should be created?"))

for font in fonts:
    for number in range(0,10):
        folder_path = f"/Users/tobiasschulz/Documents/GitHub/digit_dataset/printed_digits/train_images/{font}/{number}"
        # find all files paths from the folder
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        images = [ x for x in images if ".DS_Store" not in x ]

        num_generated_files = 0
        while num_generated_files <= num_files_desired:
            # random image from the folder
            image_path = random.choice(images)
            # read image as an two dimensional array of pixels
            image_to_transform = io.imread(image_path, plugin="matplotlib")
            print("Image loaded")
            # random num of transformation to apply
            num_transformations_to_apply = random.randint(1, len(available_transformations))

            num_transformations = 0
            transformed_image = None
            while num_transformations <= num_transformations_to_apply:
                # random transformation to apply for a single image
                key = random.choice(list(available_transformations))
                transformed_image = available_transformations[key](image_to_transform)
                num_transformations += 1

            new_file_path = f'%s/{font}_{number}_augmented_image_%s.jpg' % (folder_path, num_generated_files)

            # write image to the disk
            io.imsave(new_file_path, transformed_image)
            num_generated_files += 1
            print(f'{num_generated_files} files saved')