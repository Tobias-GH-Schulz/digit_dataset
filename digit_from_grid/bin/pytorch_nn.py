
#%%
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataset
import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from nn_module import Net
import torch.optim as optim
from load_data import personalMINST
import numpy as np
from PIL import Image

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
        (batch_idx*64) + ((epoch - 1)*len(train_loader.dataset)))
        #torch.save(network.state_dict(), '/results/model.pth')
        #torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# VARIABLE FOR THE TRAINING AND TESTING
train_size = 0.6
n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.019
momentum = 0.9
log_interval = 5

#LOANDING THE DATASET
#THE TRANSFORM LET YOU SET THE TRANSFORMATIONA THAT YOU WANT ON THE DATASET
transform_img = transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor()
])

#LOAD TRAIN AND TEST BY CALLING THE PERSONALMINST CLASS AND RELATIVE METHOD (you can set train_size param to change the lenght of train and test size)
dataset_l = personalMINST('digit_from_grid/final_dataset', 'digit_from_grid/final_dataset/label_final.csv',train_size = train_size, transform = transform_img)
train_loader = DataLoader(dataset=dataset_l.get_train(), batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(dataset=dataset_l.get_test(), batch_size=batch_size_test, shuffle=True)

#LOADING THE NETWORK FROM THE NN_MODULE.PY 
network = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)

#SETTING VARIABLE TO STORE LOSSES

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

#CALLING THE TEST BEFORE TRAINING TO EVALUATE OUR MODEL WITH RANDOMLY INITIALIZED PARAMETERS
test()
for epoch in range(1 , n_epochs + 1):
    train(epoch=epoch)
    test()
    print(epoch, "/", n_epochs)

torch.save(network,'digit_from_grid/sudoku_model.pt')

dataset_sudoku = DataLoader(dataset=dataset_l.get_sudoku('digit_from_grid/img_sudoku','digit_from_grid/img_sudoku/path.csv', transform = transform_img))
#dataset_sudoku = DataLoader(dataset_sudoku)
outputs = []
for data, target in dataset_sudoku:
  if np.count_nonzero(data.numpy()) != 0:
    output = network(data)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    outputs.append(prediction)
  else: outputs.append(0)

  
print("TEST IMAGE")
print(np.reshape(outputs, (9,9)))

'''
# PART TO CHECK IF TRAIN AND TEST HAS INTEGRITY BETWEEN IMAGES AND LABEL
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)

print(type(iter(train_loader)))

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig.savefig('test.jpg') #CHECK test.jpg NOT plt.show() compatibility issue for testing 

#print(features_data.shape, label_data.shape)
#print(len(label_data))

'''

