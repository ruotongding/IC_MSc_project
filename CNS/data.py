import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset
from tqdm import tqdm

class NPYSliceDataset(Dataset):
    def __init__(self, data_folder, autoencoder):
        self.data_folder = data_folder
        self.autoencoder = autoencoder
     
        self.file_names = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')],
                                 key=lambda x: int(re.findall(r'\d+', x)[0]))

    # 初始化时加载所有文件、转换为张量并进行编码，显示进度条
        self.file_names = self.file_names

        self.encoded_data_list = []
        self.data_tensor_list = []
        with torch.no_grad():
            for file_name in tqdm(self.file_names, desc="Loading and encoding data", position=0, leave=False):
                file_path = os.path.join(self.data_folder, file_name)
                data = np.load(file_path) 
                data_tensor = torch.tensor(data, dtype=torch.float32) 
                encoded = self.autoencoder.encoder(data_tensor) 
                self.encoded_data_list.append(encoded)  
                self.data_tensor_list.append(data_tensor)  
        self.total_slices = len(self.file_names) * (21 - 10)

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        file_idx = idx // (21 - 10)
        slice_idx = idx % (21 - 10)

        encoded = self.encoded_data_list[file_idx]
        input_slice = encoded[slice_idx:slice_idx + 5]  # (5, 256)
        target_slice = encoded[slice_idx + 5:slice_idx + 10]  # (5, 256)
        original_target_slice = self.data_tensor_list[file_idx][slice_idx + 5:slice_idx + 10]  # (5, 4, 128, 128)

       
        return (
        input_slice.clone().detach().requires_grad_(False),
        target_slice.clone().detach().requires_grad_(False),
        original_target_slice.clone().detach().requires_grad_(False)
    )



