import sys
import numpy as np
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as Data
import torchvision 

from modules import cheat_classifier

if __name__ == "__main__":

    X_all = torch.from_numpy(np.load("../data/X_novel_train.npy")).view(-1,3,32,32)
    Y_all = torch.from_numpy(np.load("../data/Y_novel_train.npy").astype(np.long)).view(-1,1)

    X_train = torch.zeros([20*480,3,32,32],dtype=torch.float32)
    Y_train = torch.zeros([20*480,1],dtype=torch.long)
    X_valid = torch.zeros([20*20,3,32,32],dtype=torch.float32)
    Y_valid = torch.zeros([20*20,1],dtype=torch.long)

    train_index = 0
    valid_index = 0

    for i in range(0,len(X_all),500):
        train_index = 480 * int(i / 500)
        valid_index = 20 * int(i / 500)

        X_train[train_index:(train_index+480)] = X_all[i:(i+480)]
        Y_train[train_index:(train_index+480)] = Y_all[i:(i+480)]
        X_valid[valid_index:(valid_index+20)] = X_all[(i+480):(i+500)]
        Y_valid[valid_index:(valid_index+20)] = Y_all[(i+480):(i+500)]

    EPOCH = 500
    PATIENCE = 15
    BATCH_SIZE = 128
    model = cheat_classifier().cuda()
    CELoss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))
    best_acc = 0.0
    count = 0

    for epoch in range(EPOCH):

        epoch_loss = 0.0
        acc = 0.0
        start = datetime.datetime.now()

        random_index = np.random.permutation(len(Y_train))
        x_train_shuffle = X_train[random_index]
        y_train_shuffle = Y_train[random_index].squeeze()

        for i in range(0,len(X_train),BATCH_SIZE):

            model.train()
            
            if (i + BATCH_SIZE > len(X_train)):
                input_X = x_train_shuffle[i:]
                input_Y = y_train_shuffle[i:]
            else:
                input_X = x_train_shuffle[i:(i + BATCH_SIZE)]
                input_Y = y_train_shuffle[i:(i + BATCH_SIZE)]
            
            _fv,output = model(input_X.cuda())
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
            
            _fv,output = model(input_X.cuda())
            batch_loss = CELoss(output,input_Y.squeeze().cuda())
            val_loss += batch_loss.item()

            #print (torch.argmax(output, 1).cpu(),input_Y)
            val_acc += torch.sum((torch.argmax(output, 1).cpu() == input_Y.squeeze()))
            progress = ('=' * int(40 * i / len(X_valid)) ).ljust(40)
            print ('[%02d/%02d] %2.4f sec(s) | %s | %d%%' % (epoch+1, EPOCH, (datetime.datetime.now() - start).total_seconds(), progress, int(i / len(X_train) * 100)), end='\r', flush=True)

        print('[%02d/%02d] %6.4f sec(s) Train: %.4f/ %.4f      Val: %.4f/ %.4f' % (epoch+1, EPOCH,(datetime.datetime.now() - start).total_seconds(), epoch_loss, float(acc)/len(X_train), val_loss, float(val_acc)/len(X_valid)))
        if (val_acc > best_acc):
            count = 0
            best_acc = val_acc
            torch.save(model, '../models/pre_trained_cheat.pkl')
        else:
            count += 1
        if (count == PATIENCE):
            break


    X_test = torch.from_numpy(np.load("../data/X_novel_test.npy")).view(-1,3,32,32)
    table = np.load("../data/class_table.npy")
    model = torch.load('../models/pre_trained_cheat.pkl')

    _fv,predict = model(X_test.cuda())

    f = open("cheat.csv","w")
    f.write('image_id,predicted_label\n')
    for i in range(2000):
        index = torch.argmax(predict[i])
        f.write(str(i) + "," + str(table[index])+"\n")
    f.close()    

