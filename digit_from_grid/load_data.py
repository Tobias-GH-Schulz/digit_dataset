import os
import torch
#from tensorflow.keras.preprocessing import image
from torch.utils.data import Dataset, DataLoader, dataset
from torchvision import transforms
import PIL
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class generateCSV():
    def __init__(self, path_folder):
        self.list_file = os.listdir(path_folder)
        self.list_file.sort()
        temp = []
        
        for files in self.list_file:
           dict_t = {
               'path':path_folder + files, 
                'label':0
           }
           temp.append(dict_t)
        print(self.list_file)
        
        df = pd.DataFrame(data = temp, columns=['path', 'label'])
        #print(df)
        df.to_csv(path_folder + "path.csv", index=None)

class personal_dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform = None):
        self.root = root_dir
        self.image_dir = os.listdir(root_dir)
        self.images_files = []
        self.paths = pd.read_csv(csv_file).iloc[:, 0]
        self.data = pd.read_csv(csv_file).iloc[:, 1]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.paths[index])
        label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return (image, label)
        
#CLASS TO LOAD FROM THE PERSONAL DATASET THE TRAIN AND TEST    
class personalMINST():
    def __init__(self, path, csv_path, train_size = 0.8, transform = None):
        self.dataset = personal_dataset(path, csv_path, transform=transform)    
        lenghts = [int(len(self.dataset) * train_size) + 1, int(len(self.dataset) * (1 - train_size))]
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, lenghts)

    def get_train(self):
        return self.train_dataset

    def get_test(self):
        return self.test_dataset

    def get_sudoku(self, path, csv_path, transform = None):
        self.dataset_sudoku = personal_dataset(path, csv_path, transform=transform)  
        return self.dataset_sudoku


'''
temp = generateCSV('digit_from_grid/img_r/sudoku_grid/')
#JUNK CODE FOR DEV




def extract_label(string):
   ret_string = string.split('NUMBER_')[1]
   return ret_string.split('_')[0]

def load_images_to_data(path_folder, test_train = 'train'):
    features_data = np.empty((1,28,28,1))
    label_data = np.empty((1))
    folder_names = os.listdir(path_folder)
    for folder in folder_names:
        print("Loading: ", folder, end='[')
        image_base_path = os.path.join(path_folder, folder)
        file_names = os.listdir(image_base_path) 
        for file_n, i in zip(file_names, range(1,len(file_names),1)):
            file_base_path = os.path.join(image_base_path, file_n)
            if ".jpg" in file_n:
                img = Image.open(file_base_path).convert("L")
                img = np.resize(img, (28,28,1))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,28,28,1)
                features_data = np.append(features_data, im2arr, axis=0)
                label_data = np.append(label_data, extract_label(file_n))
            if i % 100 == 0: print("-",end='')
        print("]")


temp = personalMINST('img_aug', 'path_label.csv')
print(len(temp))
lenghts = [int(len(temp) * 0.8) + 1,int(len(temp) * 0.2)]
print(lenghts)
train_dataset, test_dataset = torch.utils.data.random_split(temp, lenghts)

def show_image(image, label, dataset):
    print(f"Label: {label}") 
    plt.imshow(image)
    plt.savefig('test.jpg')

show_image(*train_dataset[0], train_dataset)


'''

