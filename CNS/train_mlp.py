
import os
import torch
from tqdm import tqdm
import numpy as np
import torch
import random
import re
from torch.utils.data import Dataset
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

from model import KANLinear, KAN, GRUModel, LSTMModel, MLP,PowerfulAutoencoder
from data_all import NPYSliceDataset


#define model save path
content_path = '..'
model_save_path = content_path + '/model/mlp.pth'
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
print(torch.__version__)
print(torch.version.cuda)








AE_path = content_path + '/model/AE_CFD_KAN_2_channels.pth'
latent_dim = 256
Autoencoder = PowerfulAutoencoder(latent_dim=latent_dim) #use cpu
Autoencoder.load_state_dict(torch.load(AE_path))
Autoencoder = Autoencoder.cpu()
Autoencoder.eval()  # Set the model to evaluation mode

data_path = content_path+'/CFD_2_channels'
dataset = NPYSliceDataset(data_path, Autoencoder)






class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

seed = 5
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# Create DataLoader
batch_size = 32

# Split the dataset into training and testing sets (80% training, 20% testing)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers =0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = 0)

criterion = nn.MSELoss()

input_dim = 5 * 256
output_dim = 5 * 256
input_dims = [input_dim,1024,output_dim]  # Example hidden layers
#define models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dims,base_activation=nn.Identity()).to(device)
RNN_flag = False
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=10, min_delta=1e-5, path=model_save_path)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

beta = 0.01
# Training loop
num_epochs =30
train_losses = []
test_losses = []
stop_epochs = num_epochs
print("model:\n")
print(model)
print("begin training")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    step = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",position = 0, leave = False):
        step+=1
        inputs, targets, _ = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        #mlp and KAN
        if RNN_flag == False:
          flatten_inputs = inputs.reshape(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
          flatten_outputs = model(flatten_inputs)
          outputs = flatten_outputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2])
        elif RNN_flag == True:
          outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    #validation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            if RNN_flag == False:

              flatten_inputs = inputs.reshape(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
              flatten_outputs = model(flatten_inputs)

              outputs = flatten_outputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2])


            elif RNN_flag == True:
              outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    scheduler.step(test_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    #Check early stopping condition
    early_stopping(test_loss,model)
    if early_stopping.early_stop:

        print("Early stopping")
        print("best_loss",early_stopping.best_loss)
        break


print('Training complete')
train_losses_array = np.array(train_losses)
test_losses_array = np.array(test_losses)
epochs = len(train_losses_array)

# Create an array of epoch numbers
epoch_numbers = np.arange(1, epochs + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epoch_numbers, train_losses_array, label='Training Loss')
plt.plot(epoch_numbers, test_losses_array, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs of MLP ')
plt.legend()

# Save the plot
plt.savefig('results/loss_curve_mlp.png')



