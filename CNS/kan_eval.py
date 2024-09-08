

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
from data import NPYSliceDataset


#define model save path
content_path = '..'
model_save_path = content_path + '/model/kan.pth'
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






# Create DataLoader
batch_size = 32
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers = 0)

criterion = nn.MSELoss()

input_dim = 5 * 256
output_dim = 5 * 256
input_dims = [input_dim,1024,output_dim]  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(input_dims).to(device)

RNN_flag = False

# Define the function to calculate RRMSE
def rrmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2)) / np.sqrt(np.mean(img1 ** 2))

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=0,win_size = 3)
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Initialize lists to store RRMSE and SSIM scores
rrmse_scores = []
ssim_scores = []
mse_scores = []
psnr_values = []

# Ensure the model is in evaluation mode
model.load_state_dict(torch.load(model_save_path))
print(model)
model.eval()
Autoencoder.to(device)
Autoencoder.eval()
slice_steps = 5
N=128

with torch.no_grad():
    total_rrmse = 0
    total_ssim = 0
    total_psnr = 0
    total_mse = 0
    global_idx = 0
    count = 0
    global_idx = 0
    for idx, data in enumerate(test_loader):
        inputs, targets,ori_target = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)
        if RNN_flag == False:
          flatten_inputs = inputs.reshape(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
          flatten_outputs = model(flatten_inputs)
          outputs_encoded = flatten_outputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2])


        elif RNN_flag == True:
          outputs_encoded = model(inputs)
        batch_rrmse = 0
        batch_ssim = 0
        batch_mse = 0
        batch_psnr = 0
        decoded_sequence = []

        # Iterate over each time step
        for t in range(outputs_encoded.shape[1]):  
            encoded_step = outputs_encoded[:, t, :]  # Shape: (batch_size, 256)
            decoded_step = Autoencoder.decoder(encoded_step)  # Shape: (batch_size, 2, 128, 128)
            decoded_sequence.append(decoded_step)
            # Compare with the corresponding original target
            ori_step = ori_target[:, t, :, :, :]  # Shape: (batch_size, 2, 128, 128)

            # Convert tensors to numpy arrays for calculation
            decoded_step_np = decoded_step.cpu().numpy()
            ori_step_np = ori_step.cpu().numpy()

            # Calculate RRMSE and SSIM for the current step
            for i in range(batch_size):
                batch_rrmse += rrmse(ori_step_np[i], decoded_step_np[i])
                batch_ssim += compute_ssim(ori_step_np[i], decoded_step_np[i])
                batch_psnr+=compute_psnr(ori_step_np[i], decoded_step_np[i])
                batch_mse+=np.mean((ori_step_np[i]- decoded_step_np[i]) ** 2)
                #visualize:
                if i == 1 or i ==2:
                    # Plot the original and reconstructed images for each channel
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows and 2 columns

                    for c in range(2):  # Two channels
                        # Original image for Channel c
                        axes[0, c].imshow(ori_step_np[i, c, :, :], cmap='viridis')
                        axes[0, c].set_title(f'Original - Channel {c+1}')
                        axes[0, c].axis('off')

                        # Reconstructed image for Channel c
                        axes[1, c].imshow(decoded_step_np[i, c, :, :], cmap='viridis')
                        axes[1, c].set_title(f'Reconstructed - Channel {c+1}')
                        axes[1, c].axis('off')
                                        
                    plt.tight_layout()
                    plt.savefig(f'results/pair_comparison_sample_{i}_step_{t}.png')
                    plt.close(fig)



        decoded_sequence = torch.stack(decoded_sequence, dim=1)  # Shape: (batch_size, 5, 2, 128, 128)
        print("decoded_seq.shape", decoded_sequence.shape)
        print("ori_target.shape",ori_target.shape)

        if idx == 0:  # Visualize the first batch
            for i in range(min(2, batch_size)):  # Visualize one or two pairs
                fig, axes = plt.subplots(4, 5, figsize=(15, 12))  
                
                for t in range(5):  # For each time step
                    for c in range(2):  # For each channel
                        # Original
                        axes[2*c, t].imshow(ori_target[i, t, c].cpu().numpy(), cmap='viridis')
                        axes[2*c, t].set_title(f'Original - Step {t+1}, Channel {c+1}')
                        axes[2*c, t].axis('off')
                        
                        # Reconstructed
                        axes[2*c+1, t].imshow(decoded_sequence[i, t, c].cpu().numpy(), cmap='viridis')
                        axes[2*c+1, t].set_title(f'Reconstructed - Step {t+1}, Channel {c+1}')
                        axes[2*c+1, t].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'results/reconstruction_vs_original_sample_{i}.png')
                plt.close(fig)

        # Normalize by the number of time steps
        batch_rrmse /= (batch_size * outputs_encoded.shape[1])
        batch_ssim /= (batch_size * outputs_encoded.shape[1])
        batch_psnr/=(batch_size * outputs_encoded.shape[1])
        batch_mse/=(batch_size * outputs_encoded.shape[1])

        # Accumulate for the entire dataset
        total_rrmse += batch_rrmse
        total_ssim += batch_ssim
        total_psnr+=batch_psnr
        total_mse+=batch_mse
        count += 1

    # Calculate average RRMSE and SSIM over all batches
    avg_rrmse = total_rrmse / count
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr/count
    avg_mse = total_mse/count

    print(f'Average RRMSE: {avg_rrmse}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Average MSE: {avg_mse}')
    print(f'Average PSNR: {avg_psnr}')



