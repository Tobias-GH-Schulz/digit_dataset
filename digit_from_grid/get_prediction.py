import torch
from torch.utils.data import DataLoader
from bin.load_data import personalMINST
import numpy as np
from torchvision import transforms

class getPrediction():
    def __init__(self):
        self.model = torch.load('digit_from_grid/sudoku_model.pt')
    

    
    def get_predictions(self, data):
        return self.model(data)

transform_img = transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor()
])

dataset_l = personalMINST('digit_from_grid/final_dataset', 'digit_from_grid/final_dataset/label_final.csv',train_size = 0.8, transform = transform_img)
dataset_sudoku = DataLoader(dataset=dataset_l.get_sudoku('digit_from_grid/img_sudoku','digit_from_grid/img_sudoku/path.csv', transform = transform_img))

model = getPrediction()


outputs = []
print(dataset_sudoku.shape)
for data, target in dataset_sudoku:
  print(data.shape)
  if np.count_nonzero(data.numpy()) != 0:
    output = model.get_predictions(data)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    outputs.append(prediction)
  else: outputs.append(0)

  
print("TEST IMAGE")
print(np.reshape(outputs, (9,9)))