from glob import glob
import os
import shutil
import sys
from time import time
import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
import torchvision.transforms as tfms
from torchvision.transforms.functional import affine

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from collections import defaultdict
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.optim as optim
import nibabel as nib
from nilearn.image import resample_img

import argparse






class NiiDataset(Dataset):
    def __init__(self, img_path, tgt_path):
        # load all nii handle in a list
        img_dir = [i for i in os.listdir(img_path) if i[-3:] == "nii"]
        tgt_dir = [i for i in os.listdir(tgt_path) if i[-3:] == "nii"]
        
        self.images_list = []
        #self.transforms = tfms.Normalize((0.5,), (0.5,))

        
        #self.crop = tfms.CenterCrop((30, 40));
        
        for image_path in img_dir:
            tens = self.to_tensor(img_path + '/' + image_path)
            
            for j in range(tens.shape[2]):
                self.images_list.append((tens[:,:,j][None, ...]))
                #print(tens[:,:,j][None, ...].shape)
            
        self.target_list = []
        
        for image_path in tgt_dir:
            tens = self.to_tensor(tgt_path + '/' + image_path)
            
            for j in range(tens.shape[2]):
                self.target_list.append((tens[:,:,j][None, ...]))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        
        classes = torch.cat([self.target_list[idx] == 0, self.target_list[idx] == 1], 0)
        #return self.transforms((self.images_list[idx], classes))
        return (self.images_list[idx], classes)
    
    def to_tensor(self, pth):
        return torch.from_numpy(np.asarray(nib.load(pth).dataobj))



def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    target = target.type_as(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, metrics

def string_metrics(metrics, epoch_samples):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.3f}".format(k, metrics[k] / epoch_samples))

    return (", ".join(outputs))
    
def string_time(elapsed):
    return "%im %is" %(int(elapsed / 60), int(elapsed % 60))


class UNetOld(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, depth=3, scale_deg=1):
        super(UNetShort, self).__init__()

        self.layers = depth
        self.down_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()

        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scale_deg = scale_deg

        self.downscale_map = []
        self.upscale_map = []

        l = self.layers
        while l > 0:
            step = 1
            if l >= self.scale_deg:
                step = self.scale_deg

            l -= step
            self.upscale_map.append(2 ** step)

        self.downscale_map = self.upscale_map[::-1]

        features = init_features
        self.down_modules.append(self._doubleconv(in_channels, features))

        for step in self.downscale_map:
            self.downscale.append(nn.MaxPool2d(kernel_size=step, stride=step))
            self.down_modules.append(self._doubleconv(features, features * step))
            features *= step

        self.middle = self.down_modules[-1]

        for step in self.upscale_map:
            self.upscale.append(nn.ConvTranspose2d(features, features // step, kernel_size=step, stride=step))
            features //= step
            self.up_modules.append(self._doubleconv(features * 2, features))
            

        self.out = nn.Conv2d(features, out_channels, kernel_size=1)


    def _doubleconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # or 0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # or 0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        

    def forward(self, X):
        #down_outputs = [[] for i in range(self.layers)]
        
        #print(self.upscale_map)

        for i in range(self.layers):
            X = self.down_modules[i](X)
            
            #print("down", i, X.shape)
            if i in self.concat_map:
                down_outputs[i] = (X)

            X = self.pool(X)

        

        for i in range(len(self.downcale_map)):
            X = self.down_modules[i](X)
            down_outputs.append(X)
            X = self.downscale[i](X)

        X = self.middle(X)
        
        for i in range(len(self.upscale_map)):
            X = self.upscale[i](X)
            X = torch.cat((X, down_outputs[-i - 1]), dim=1)
            X = self.up_modules[i](X)

        X = self.out(X)

        return X


       
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, depth=3, scale_deg=1):
        super(UNet, self).__init__()

        self.layers = depth
        self.down_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()

        self.scale_deg = scale_deg

        self.downscale_map = []
        self.upscale_map = []

        l = self.layers
        while l > 0:
            step = 1
            if l >= self.scale_deg:
                step = self.scale_deg

            l -= step
            self.upscale_map.append(2 ** step)

        self.downscale_map = self.upscale_map[::-1]

        features = init_features
        self.down_modules.append(self._doubleconv(in_channels, features))

        for step in self.downscale_map:
            self.downscale.append(nn.MaxPool2d(kernel_size=step, stride=step))
            self.down_modules.append(self._doubleconv(features, features * step))
            features *= step

        self.middle = self.down_modules[-1]

        for step in self.upscale_map:
            self.upscale.append(nn.ConvTranspose2d(features, features // step, kernel_size=step, stride=step))
            features //= step
            self.up_modules.append(self._doubleconv(features * 2, features))
            

        self.out = nn.Conv2d(features, out_channels, kernel_size=1)


    def _doubleconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # or 0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # or 0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        

    def forward(self, X):
        down_outputs = []
        
        for i in range(len(self.downscale_map)):
            X = self.down_modules[i](X)
            down_outputs.append(X)
            X = self.downscale[i](X)

        X = self.middle(X)
        
        for i in range(len(self.upscale_map)):
            X = self.upscale[i](X)
            X = torch.cat((X, down_outputs[-i - 1]), dim=1)
            X = self.up_modules[i](X)

        X = self.out(X)

        return X