from sys import path
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import save_img
import pandas as pd
import os


font_list = ['Helvetica','Calibri','Futura','Garamond','TimesNewRoman','Arial','Cambria','Verdana','Rockwell']
string_path =  'digit_from_grid/img_t/NUMBER_{}_{}.jpg'
img_arr = {'Helvetica':[],'Calibri':[],'Futura':[],'Garamond':[],'TimesNewRoman':[],'Arial':[],'Cambria':[],'Verdana':[],'Rockwell':[]}
for i in font_list:
    for j in range(1,10,1):
        print(string_path.format(i,j))
        img_arr[i].append(img_to_array(load_img(string_path.format(i,j))))
        

#PATH_NAME OF NEW GENERATED IMAGES
path_save = '{}/{}-NUMBER_{}_aug0.{}.jpg'
#FOLDER WHERE GENERATED IMAGES ARE STORED
fold_path = 'digit_from_grid/img_aug/'
#NUMBER OF IMAGES GENERATED
n_images_generated = 124 #number of images generated
list_ret = []
for data in font_list:
    #IF FOLDER EXIST COMMENT THIS LINE
    #os.mkdir(fold_path + "{}".format(data))
    for num, i in zip(img_arr[data], range(1,10,1)):
        samples = expand_dims(num, 0)
        # VARY THOSE PARAMETERS TO GET DIFFERENT AUGMENTATIONS
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=30,  #rotation_range set to 30
            width_shift_range=0.1, # widht shift_range set to 0.2
            height_shift_range=0.1, # height shift_range set to 0.2
            brightness_range= [0.7, 1], #brightness_range set [0.7, 1]
            shear_range=0.0,
            zoom_range= [0.7, 1], #zoom variations set [0.7, 1]
            channel_shift_range=0.0,
            fill_mode="nearest",
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0,
            dtype=None,
        )
        it = datagen.flow(samples, batch_size=1)
        for j in range(n_images_generated):
            batch = it.next()
            temp_dict = {
                "path": fold_path + path_save.format(data,data, i,j),
                "label": i
            }
            list_ret.append(temp_dict)
            print("PATH_SAVE ", fold_path + path_save.format(data,data, i,j + 1))
            save_img(fold_path + path_save.format(data,data,i,j), batch[0])

df = pd.DataFrame(list_ret, columns=['path', 'label'])
df.to_csv('path_label.csv', index = False)
print("AUGMENTATION COMPLETED")
