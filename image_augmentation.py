from sys import path
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import save_img
import os
#from matplotlib import pyplot


font_list = ['Helvetica','Calibri','Futura','Garamond','TimesNewRoman','Arial','Cambria','Verdana','Rockwell']
string_path =  'img_s/{}-NUMBER_{}.jpg'
img_arr = {'Helvetica':[],'Calibri':[],'Futura':[],'Garamond':[],'TimesNewRoman':[],'Arial':[],'Cambria':[],'Verdana':[],'Rockwell':[]}
for i in font_list:
    for j in range(1,10,1):
        print(string_path.format(i,j))
        img_arr[i].append(img_to_array(load_img(string_path.format(i,j))))
        #print(img_arr[i])
counter = 1
counter_two = 0
temp_count = 0
#os.mkdir("img_aug")
    #print("img_aug/{}".format(i))

path_save = 'img_aug/{}/{}-NUMBER_{}_aug0.{}.jpg'

for data in font_list:
    os.mkdir("img_aug/{}".format(data))
    for num, i in zip(img_arr[data], range(1,10,1)):
        samples = expand_dims(num, 0)
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=30,  #rotation_range set to 30
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=0.0,
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
        for j in range(9):
            batch = it.next()
            print("PATH_SAVE ", path_save.format(data,data, i,j))
            save_img(path_save.format(data,data,i,j), batch[0])
            #if i == 7: break
    #if temp_count == 30:
    #    break

print(len(img_arr))
#datagen = ImageDataGenerator()
