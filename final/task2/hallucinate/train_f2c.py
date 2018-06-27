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

from modules import feature2classes

if __name__ == '__main__':
    model = feature2classes().cuda()
    EPOCH = 500
    PATIENCE = 30
    BATCH_SIZE = 128

    CELoss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999))
    best_acc = 0.0
    count = 0

    X_train = torch.from_numpy(np.load("../data/x_feature_train.npy").astype(np.float32)).view(-1,512)
    Y_train = torch.from_numpy(np.load("../data/y_feature_train.npy").astype(np.long)).view(-1)
    X_valid = torch.from_numpy(np.load("../data/x_feature_valid.npy").astype(np.float32)).view(-1,512)
    Y_valid = torch.from_numpy(np.load("../data/y_feature_valid.npy").astype(np.long)).view(-1)

    print (Y_train[-1],Y_valid[-1])

    for epoch in range(EPOCH):
        epoch_loss = 0.0
        acc = 0.0
        start = datetime.datetime.now()

        random_index = np.random.permutation(len(Y_train))
        x_train_shuffle = X_train[random_index]
        y_train_shuffle = Y_train[random_index]

        for i in range(0,len(X_train),BATCH_SIZE):

            model.train()
            
            if (i + BATCH_SIZE > len(X_train)):
                input_X = x_train_shuffle[i:]
                input_Y = y_train_shuffle[i:]
            else:
                input_X = x_train_shuffle[i:(i + BATCH_SIZE)]
                input_Y = y_train_shuffle[i:(i + BATCH_SIZE)]
            
            output = model(input_X.cuda())
            batch_loss = CELoss(output,input_Y.cuda())

            model.zero_grad()
            batch_loss.backward()
            opt.step()

            epoch_loss += batch_loss.item()
            #print (torch.argmax(output, 1).cpu(),input_Y)
            acc += torch.sum((torch.argmax(output, 1).cpu() == input_Y))
            progress = ('=' * int(40 * i / len(X_train)) ).ljust(40)
            print ('[%02d/%02d] %2.4f sec(s) | %s | %d%%' % (epoch+1, EPOCH, (datetime.datetime.now() - start).total_seconds(), progress, int(i / len(X_train) * 100)), end='\r', flush=True)

        #print ("\nEpoch: ",epoch+1, "Acc: ", acc.item()/len(X_train), "\n")

        val_loss = 0.0
        val_acc = 0.0

        for i in range(0,len(X_valid),BATCH_SIZE):

            model.eval()
            
            if (i + BATCH_SIZE > len(X_valid)):
                input_X = X_valid[i:]
                input_Y = Y_valid[i:]
            else:
                input_X = X_valid[i:(i + BATCH_SIZE)]
                input_Y = Y_valid[i:(i + BATCH_SIZE)]
            
            output = model(input_X.cuda())
            batch_loss = CELoss(output,input_Y.cuda())
            val_loss += batch_loss.item()

            #print (torch.argmax(output, 1).cpu(),input_Y)
            val_acc += torch.sum((torch.argmax(output, 1).cpu() == input_Y))
            progress = ('=' * int(40 * i / len(X_valid)) ).ljust(40)
            print ('[%02d/%02d] %2.4f sec(s) | %s | %d%%' % (epoch+1, EPOCH, (datetime.datetime.now() - start).total_seconds(), progress, int(i / len(X_train) * 100)), end='\r', flush=True)

        print('[%02d/%02d] %6.4f sec(s) Train: %.4f/ %.4f      Val: %.4f/ %.4f' % (epoch+1, EPOCH,(datetime.datetime.now() - start).total_seconds(), epoch_loss, float(acc)/len(X_train), val_loss, float(val_acc)/len(X_valid)))
        if (val_acc > best_acc):
            count = 0
            best_acc = val_acc
            torch.save(model, '../models/FV_trained.pkl')
        else:
            count += 1
        if (count == PATIENCE):
            break