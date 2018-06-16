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

from modules import base_classifier, cheat_classifier

if __name__ == "__main__":

    X_train = torch.from_numpy(np.load("../data/X_novel_train.npy")).view(-1,3,32,32)
    Y_train = torch.from_numpy(np.load("../data/Y_novel_train.npy").astype(np.long)).view(-1)
    X_test_img = torch.from_numpy(np.load("../data/X_novel_test.npy")).view(-1,3,32,32)
    table = np.load("../data/class_table.npy")

    model = cheat_classifier()
    model_dict = model.state_dict()

    base_model = torch.load("../models/pre_trained_39.pkl").cuda()
    base_dict = base_model.state_dict()
    pre_dict = {k: v for k, v in base_dict.items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    X_support = torch.zeros([20,8192],dtype=torch.float32)
    Y_support = torch.zeros([20,1],dtype=torch.long)

    for i in range(20):
        input_X = X_train[(i*500):(i*500+10)]
        input_Y = Y_train[(i*500)].view(-1,1)
        latent_X,_output = model(input_X.cuda())
        X_support[i] = torch.mean(latent_X,0)
        Y_support[i] = input_Y

    X_test,_output = model(X_test_img.cuda())
    predict = list()

    for i in range(len(X_test)):
        dis = 2000000000
        index = 0

        for j in range(20):
            #print (X_support[j])
            tmp = torch.sum((X_test[i].cpu() - X_support[j]).abs()).item()
            #print (tmp)
            if (tmp < dis):
                dis = tmp
                index = table[j]

            predict.append(index)

        #print (index)

    f = open("output.csv","w")
    f.write('image_id,predicted_label\n')
    for i in range(2000):
        f.write(str(i) + "," + str(predict[i]+"\n"))
    f.close()

    """

    EPOCH = 500
    PATIENCE = 20
    BATCH_SIZE = 128
    model = base_classifier().cuda()
    CELoss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=0.001)
    best_acc = 0.0
    count = 0

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
            torch.save(model, '../models/pre_trained_2.pkl')
        else:
            count += 1
        if (count == PATIENCE):
            break
    """


