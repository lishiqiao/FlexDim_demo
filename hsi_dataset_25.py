from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import os
from scipy.io import loadmat
import hdf5storage
import h5py
import torch
import torch.nn.functional as F
from utils import angle_supperresolution,Fast_rFFT2d_GPU_batch,to_lightfield,hyperspectral_to_rggb,lightfield_resize,to_view

device = torch.device('cuda')


def z_score_normalize(x: torch.Tensor) -> torch.Tensor:

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)

    std = torch.max(std, torch.tensor(1e-8, device=x.device))

    return (x - mean) / std

class TestDataset(Dataset):
    def __init__(self, data_root,bgr2rgb=True):
        self.hypers = []
        self.bgrs = []

        hyper11_data_path = f'{data_root}/11_test/'
        hyper5_data_path = f'{data_root}/5_test/'
        hyper11_list = sorted([f for f in os.listdir(hyper11_data_path) if f.endswith('Tao_5_11.mat')])
        hyper5_list = sorted([f for f in os.listdir(hyper5_data_path) if f.endswith('Tao_5_25.mat')])  # GT
        print(f'len(hyper) of dataset: {len(hyper11_list)}')
        for i in range(len(hyper11_list)):
            hyper11_path = os.path.join(hyper11_data_path, hyper11_list[i])
            hyper5_path = os.path.join(hyper5_data_path, hyper5_list[i])
            hyper11 = hdf5storage.loadmat(hyper11_path)
            hyper11 = hyper11['data'][:] / 120000.
            hyper11 = torch.tensor(hyper11, dtype=torch.float16).to(device)

            rgb11 = hyperspectral_to_rggb(hyper11)
            rgb11 = rgb11.unsqueeze(0)  # C,h,w
            rgb11 = to_view(rgb11, 11, 11)
            rgb11 = rgb11.squeeze(0)
            C, ch, ch, H, W = rgb11.shape
            rgb11 = rgb11.reshape(C, 121, H, W)  # 1,3,11,11,595,515
            rgb11 = rgb11 / 7


            # GT
            hyper5 = hdf5storage.loadmat(hyper5_path)
            hyper5 = hyper5['data'] / 65535.
            hyper5 = torch.tensor(hyper5, dtype=torch.float16)
            hyper5 = hyper5.permute(2, 0, 1)
            hyper5 = to_view(hyper5.unsqueeze(0), 5, 5)
            hyper5 = hyper5.squeeze(0)
            C, ch, ch, H, W = hyper5.shape
            hyper5 = hyper5.reshape(C, ch * ch, H, W)
            print(hyper5.max())
            hyper5 = hyper5.to(device)

            self.hypers.append(hyper5[:28, :, :].cpu())
            self.bgrs.append(rgb11.cpu())



    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]

        # filters = self.filters
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)


