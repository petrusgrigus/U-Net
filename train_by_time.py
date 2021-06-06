from source import *

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



    
############## params ########################
# train_unet spleen -d 3 1 5 

parser = argparse.ArgumentParser()
parser.add_argument('--prob', '-p', help="chosen dataset", type=str, default='spleen')
parser.add_argument('--dep', '-d', help="unet depth", type=int, default=4)
parser.add_argument('--scl', '-s', help="unet scale", type=int, default=1)
parser.add_argument('--time', '-t', help="maximum training time", type=int, default=5)
parser.add_argument('--lr', '-l', help="learning rate", type=float, default=1e-2)
parser.add_argument('--enum', '-e', help="maximum epoch number", type=int, default=100)

args = parser.parse_args()

TASK = args.prob
DEPTH = args.dep
SCALE = args.scl
MAX_TIME = args.time
NUM_EPOCHS = args.enum
START_LR = args.lr

params = {
    'in_channels':1, 
    'out_channels':2, 
    'depth':DEPTH, 
    'init_features':16,
    'scale_deg':SCALE
}

print(sys.argv)
    
name = "unet-" + str(DEPTH) + "x" + str(SCALE) + "-" + str(MAX_TIME)
if START_LR != 1e-2:
    name += '-lr=' + str(START_LR)

if TASK != 'lung':
    DIR = './spleen'
    task = '/large/home/pvgrigorev/Spleen/Task09_Spleen'
else:
    DIR = './lung'
    task = '/large/home/pvgrigorev/Lungs/Task06_Lung'
    
############## params ########################

logs = open(DIR + '/logs/' + name + ".txt", 'w')

if os.path.exists(DIR + '/graphs/' + name + "/"):
    shutil.rmtree(DIR + '/graphs/' + name + "/")

os.mkdir(DIR + '/graphs/' + name + "/")
trainf = DIR + '/graphs/' + name + "/" + name + "-train"
valf = DIR + '/graphs/' + name + "/" + name + "-val"



data_dir = task + '/imagesTr'
labels_dir = task + '/labelsTr'
batch_size = 10

full_dataset = NiiDataset(data_dir, labels_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_PRINT = len(train_dataloader) // 5

def train_model(model, optimizer, scheduler, num_epochs=25):
    global train, val
    start = time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('\n         [ U %i x %i ]         ' %(DEPTH, SCALE), end = '')
        print('\n======== Epoch %i / %i ========' %(epoch + 1, num_epochs))
        print('\n======== Epoch %i / %i ========' %(epoch + 1, num_epochs), file=logs)
        print('Training...')
        print('Training...', file=logs)
        
        # =========== training ==========

        metrics = defaultdict(float)
        epoch_samples = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if step % BATCH_PRINT == 0 and not step == 0:
                print('  Batch %i  of  %i;\tElapsed time: ' %(step + 1, len(train_dataloader)) + string_time(time() - start))
                print('  Batch %i  of  %i;\tElapsed time: ' %(step + 1, len(train_dataloader)) + string_time(time() - start), file=logs)
                #print('      ' + str(step + 1) + " batch training loss: "  + string_metrics(metrics, epoch_samples))
                
            optimizer.zero_grad()
            
            #print(inputs.shape, labels.shape, )#outputs.shape)
            outputs = model(inputs)
            
            
            loss, metrics = calc_loss(outputs, labels, metrics)
            epoch_samples += inputs.size(0)
            
            
            
            loss.backward()
            optimizer.step()
            # del outputs
            # del inputs
            # del labels
            # del loss
        
        train.append([time() - start, metrics['bce'] / epoch_samples, metrics['dice'] / epoch_samples])
        print("\n  Average training loss: "  + string_metrics(metrics, epoch_samples))
        print("\n  Average training loss: "  + string_metrics(metrics, epoch_samples), file=logs)
        scheduler.step()
        
        # =========== validating ===========
        
        print("\nValidating...")
        print("\nValidating...", file=logs)
        
        metrics = defaultdict(float)
        epoch_samples = 0
        model.eval()
        
        for step, batch in enumerate(val_dataloader):
          
            inputs, labels = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss, metrics = calc_loss(outputs, labels, metrics)

            epoch_samples += inputs.size(0)
            
            # del outputs
            # del inputs
            # del labels
            # del loss

        epoch_loss = metrics['loss'] / epoch_samples
        
        # ============= logging ==============
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss


        val.append([time() - start, metrics['bce'] / epoch_samples, metrics['dice'] / epoch_samples])
        print("\n  Average validation loss: "  + string_metrics(metrics, epoch_samples))
        print("\n  Average validation loss: "  + string_metrics(metrics, epoch_samples), file=logs)
        
        print("  Elapsed time: " + string_time(time() - start))
        print("  Elapsed time: " + string_time(time() - start), file=logs)

        if time() - start >= MAX_TIME * 60 * 60:
            break

    print('Best val loss: {:.4f}'.format(best_loss))
    print('Best val loss: {:.4f}'.format(best_loss), file=logs)
    # model.load_state_dict(best_model_wts)
    
    return model

train = []
val = []

model = UNet(**params).to(device)

optimizer_ft = optim.Adam(model.parameters(), lr=START_LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

np.save(trainf, np.array(train))
np.save(valf, np.array(val))

torch.save(model.state_dict(), DIR + "/weights/" + name + ".pt")
