import torch
from torch.utils.data import DataLoader, dataset
from  image_extractor import sudoku_extractor
import numpy as np


class getPrediction():
    def __init__(self):
        self.model = torch.load('digit_from_grid/sudoku_model.pt')

    def get_predictions(self, data):
        return self.model(data)


if __name__ == '__main__':
    model = getPrediction()
    sudoku_extract = sudoku_extractor(
        'digit_from_grid/bin1.1/sudoku_img/sudoku2.jpg')
    dataset = DataLoader(dataset = sudoku_extract.get_array_cells())
    #print(array.shape)
    outputs = []

    for data in dataset:  
    #print(data.shape)
        if np.count_nonzero(data.numpy()) != 0:
            output = model.get_predictions(data)
            prediction = int(torch.max(output.data, 1)[1].numpy())
            outputs.append(prediction)
        else: outputs.append(0)

    print("TEST IMAGE")
    print(np.reshape(outputs, (9,9)))