import random
import sys
import numpy as np
import os
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as Data
import torchvision

from torchsummary import summary
from modules import Siamese, Siamese_Ultra
from dataset import MyDataset, ValDataset, FewDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_data = MyDataset()
    #train_data = FewDataset(k=10,seed=1127)
    
    val_data = ValDataset()

    train_loader = DataLoader(train_data, batch_size=200, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=False, num_workers=0)

    EPOCH = 2000
    PATIENCE = 100
    BATCH_SIZE = 150


    # ===== Loading pre_trained model =====
    pretrained_dict = torch.load('../models/Sia_trained_70.pt')
    model = Siamese_Ultra().to(device)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = torch.load( '../models/Ultra_trained_hope.pkl').cuda()
    # ======================================

    Loss = nn.BCELoss()
    opt = optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))
    best_acc = 0.0
    count = 0

    for epoch in range(EPOCH):

        print ("Epoch: ",epoch)
        true_loss = 0.0
        true_acc = 0.0

        start = datetime.datetime.now()     
        model.train()
        
        for x1, x2, label in tqdm(train_loader):
            label = label.view(-1)
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            output = model(x1,x2)
            batch_loss = Loss(output,label)
            model.zero_grad()
            batch_loss.backward()
            opt.step()

            true_loss += batch_loss.item()
            true_acc += np.mean(np.round(output.cpu().data.numpy()) == label.cpu().data.numpy())

        # ===== validation =======
        val_true_acc = 0.0
        val_false_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            model.eval()
            for x1, x2, label in tqdm(val_loader):
                label = label.view(-1)
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                output = model(x1,x2)
                batch_loss = Loss(output,label)

                val_loss += batch_loss.item()
                val_acc += np.mean(np.round(output.cpu().data.numpy()) == label.cpu().data.numpy())

        print ("Train_acc: ",true_acc/len(train_loader), "Train_loss", true_loss)
        print ("Val_acc: ",val_acc/len(val_loader))

        if (val_acc > best_acc and epoch >= 4):
            count = 0
            best_acc = val_acc
            torch.save(model, '../models/Ultra_trained_hope.pkl')
            torch.save(model.state_dict(), '../models/Ultra_trained_hope.pt')
        else:
            count += 1
        if (count == PATIENCE):
            break

            # ======
            # 5 shot: 1125
            # 10 shot: 1127
            # =====