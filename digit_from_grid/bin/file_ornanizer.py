# %%
import os
super_base = '/home/lorenzo/Desktop/Personal_project/digit_dataset/'
base_path = super_base + 'digit_from_grid/all_train_images/'
dest_path = super_base + 'digit_from_grid/final_dataset/'
dest_path_slim = 'digit_from_grid/final_dataset/'

# %%
#convert listdir in list
list_dir_vec = []
for dir_ in os.listdir(base_path):
    list_dir_vec.append(dir_)
list_dir_vec.sort()
print(list_dir_vec)
# %%
font_list = ['Helvetica','Calibri','Futura','Garamond','TimesNewRoman','Arial','Cambria','Verdana','Rockwell','Franklin Gothic']
font_list.sort()



# %%
temp_list = []
for fold_n in font_list:
    for dir_ in list_dir_vec:
        list_file = os.listdir(base_path + dir_)
        list_file.sort()
        for file_n in list_file:
            if fold_n in file_n:
                #print("SOURCE", base_path + dir_ + '/'+ file_n) #source
                #print("DEST", dest_path + fold_n + '/' + file_n) #DEST_PATH
                os.rename(base_path + dir_ + '/'+ file_n, dest_path + fold_n + '/' + file_n)
        
# %%
def extract_label(string):
    string = string.split('_')[1]
    return string.split('_')[0]
#%%
import pandas as pd
folders = os.listdir(dest_path)
folders.sort()
array_return = []
for name_folder in folders:
    file_names = os.listdir(dest_path + name_folder)
    file_names.sort()
    for f in file_names:
        temp_dict = {
            'path' : dest_path_slim + name_folder + '/' + f,
            'label' : extract_label(f)
        }
        array_return.append(temp_dict)
        
df = pd.DataFrame(data=array_return, columns=['path', 'label'])
print(df.head(10))
# %%
df.to_csv(dest_path + 'label_final.csv', index=False)
# %%
