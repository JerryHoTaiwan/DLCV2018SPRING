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

import ResNetFeat
# import losses
from modules import base_classifier

def parse_args():
    parser = argparse.ArgumentParser(description='Main training script')
    #parser.add_argument('--traincfg', required=True, help='yaml file containing config for data')
    #parser.add_argument('--valcfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--model', default='ResNet18', help='model: ResNet{10|18|34|50}')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay')
    parser.add_argument('--step_size', default=30, type=int, help='Step size')
    parser.add_argument('--print_freq', default=10, type=int,help='Print frequecy')
    parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch', default=90, type=int, help ='Stopping epoch')
    parser.add_argument('--allow_resume', default=0, type=int)
    parser.add_argument('--resume_file', default=None, help='resume from file')
    #parser.add_argument('--checkpoint_dir', required=True, help='Directory for storing check points')
    parser.add_argument('--aux_loss_type', default='l2', type=str, help='l2 or sgm or batchsgm')
    parser.add_argument('--aux_loss_wt', default=0.1, type=float, help='loss_wt')
    parser.add_argument('--num_classes',default=80, type=float, help='num classes')
    parser.add_argument('--dampening', default=0, type=float, help='dampening')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='iters for warmup')
    parser.add_argument('--warmup_lr', default=0.01, type=int, help='lr for warmup')

    return parser.parse_args()

def get_model(model_name, num_classes):
    model_dict = dict(ResNet10 = ResNetFeat.ResNet10,
                ResNet18 = ResNetFeat.ResNet18,
                ResNet34 = ResNetFeat.ResNet34,
                ResNet50 = ResNetFeat.ResNet50,
                ResNet101 = ResNetFeat.ResNet101)
    return model_dict[model_name](num_classes, False)

def isfile(x):
    if x is None:
        return False
    else:
        return os.path.isfile(x)


if __name__ == "__main__":

    np.random.seed(10)
    params = parse_args()
    model = get_model(params.model, params.num_classes)
    model = model.cuda()

    X_train = torch.from_numpy(np.load("../data/X_base_train.npy")).view(-1,3,32,32)
    Y_train = torch.from_numpy(np.load("../data/Y_base_train.npy").astype(np.long)).view(-1)
    X_valid = torch.from_numpy(np.load("../data/X_base_test.npy")).view(-1,3,32,32)
    Y_valid = torch.from_numpy(np.load("../data/Y_base_test.npy").astype(np.long)).view(-1)

    EPOCH = 500
    PATIENCE = 30
    BATCH_SIZE = 128
    #model = base_classifier().cuda()
    CELoss = nn.CrossEntropyLoss()
    # loss_fn = losses.GenericLoss(params.aux_loss_type, params.aux_loss_wt, params.num_classes)
    opt = optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))
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
            
            output,_fv = model(input_X.cuda())
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
            torch.save(model, '../models/Res_trained_39.pkl')
        else:
            count += 1
        if (count == PATIENCE):
            break


