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

import ResNetFeat
import losses
from modules import base_classifier

def parse_args():
    parser = argparse.ArgumentParser(description='Main training script')
    #parser.add_argument('--traincfg', required=True, help='yaml file containing config for data')
    #parser.add_argument('--valcfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--model', default='ResNet10', help='model: ResNet{10|18|34|50}')
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
    model = torch.load(sys.argv[1])
    #model = get_model(params.model, params.num_classes)
    model = model.cuda()
    model.eval()

    X_train = torch.from_numpy(np.load("../data/X_base_train.npy")).view(-1,3,32,32)
    Y_train = torch.from_numpy(np.load("../data/Y_base_train.npy").astype(np.long)).view(-1)

    EPOCH = 500
    BATCH_SIZE = 512
    CELoss = nn.CrossEntropyLoss()

    opt = optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))
    best_acc = 0.0
    count = 0

    summary(model,(3,32,32))

    fv_tmp = torch.zeros([80*1000,512],dtype=torch.float32)

    val_loss = 0.0
    val_acc = 0.0

    input_X = X_train[0:BATCH_SIZE]
    _output,fv = model(input_X.cuda())
    fv = fv.cpu()

    for i in range(int(3*len(X_train)/4),int(4*len(X_train)/4),BATCH_SIZE):
            
        if (i + BATCH_SIZE > len(X_train)):
            input_X = X_train[i:]
        else:
            input_X = X_train[i:(i + BATCH_SIZE)]
            
        _output,_fv = model(input_X.cuda())
        _output = _output.cpu()
        _fv = _fv.cpu()
        print (i,len(X_train))

        if (i + BATCH_SIZE > len(X_train)):
            fv_tmp[i:] = _fv
        else:
            fv_tmp[i:(i + BATCH_SIZE)] = _fv

    fv = fv_tmp.detach().numpy()
    print (fv.shape)
    np.save("../data/X_train_fv4.npy",fv)

