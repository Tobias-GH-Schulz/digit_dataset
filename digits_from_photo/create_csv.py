import os
import random
from scipy import ndarray
import pandas as pd

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import matplotlib


input_path = "./printed_digits/dataset_without_zero/dataset_printed_class_folders/"

folders = [os.path.join(input_path, f) for f in os.listdir(input_path)]
folders = [ x for x in folders if ".DS_Store" not in x ]

for folder in folders:
    foldername = folder.split('/')[-1]
    dic = {"filename":[], "target":[]}
    numbers = [os.path.join(folder, f) for f in os.listdir(folder)]
    numbers = [ x for x in numbers if ".DS_Store" not in x ]
    for number in numbers:
        digit = number[-1]
        filenames = os.listdir(number)
        filenames = [ x for x in filenames if ".DS_Store" not in x ]
        for filename in filenames:
            dic["filename"].append(filename)
            dic["target"].append(digit)
            df = pd.DataFrame.from_dict(dic)
    df.to_csv(f"{input_path}/machine_digit_dataset-{foldername}.csv", index=False)