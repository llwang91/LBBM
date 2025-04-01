import sys
sys.path.append("/home/wll/LAS-Diffusion/")
import torch
from pathlib import Path
import numpy as np
import os
import json as js
from network.model_utils import sym_init


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if entry.endswith("voxel"):
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution,
        data_folder,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_folder=data_folder
        self.images = _list_image_files_recursively(os.path.join(data_folder,"img"))
        with open(os.path.join(data_folder,"ncount.json"),"r") as f:
            self.elatsic_tensor=js.load(f)
        # initialize dofs
        sym_init()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        front, _=os.path.splitext(os.path.split(path)[1])
        # load voxel 
        dim = (64, 64, 64)
        with open(path, 'rb') as f:
            img = np.unpackbits(np.fromfile(f, dtype=np.uint8),bitorder='little')
            img = np.reshape(img, dim, order="F")
            img = img.astype(dtype=np.float32)
        img=img*2-1

        bdr=-np.ones_like(img)
        idx=np.where(img[0,:,:]==1)
        bdr[:,idx[0],idx[1]]=1
        idx=np.where(img[:,0,:]==1)
        bdr[idx[0],:,idx[1]]=1
        idx=np.where(img[:,:,0]==1)
        bdr[idx[0],idx[1],:]=1

        return {"img":img[np.newaxis,:],
                "cond":np.array([v for v in self.elatsic_tensor[front[:-5]]][:-1],dtype=np.float32),
                "bdr":bdr[np.newaxis,:],
        } 