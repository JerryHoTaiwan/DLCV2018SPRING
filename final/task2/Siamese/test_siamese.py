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
from modules import Siamese
from dataset import MyDataset, NovelDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_max(few_shot,x,model_0,model_1,model_2):
    
    k = few_shot.size()[0]
    output = torch.zeros([20]).cuda()
    for i in range(k):
        xr = x.view(-1,3,32,32)
        xr = xr.repeat(few_shot.size()[1],1,1,1)

        f = few_shot[i].cuda()
        xr = xr.cuda()
        output += model_0(f,xr)#.squeeze()
        #output += model_1(f,xr)#.squeeze()
        #output += model_2(f,xr)#.squeeze()
    return torch.argmax(output).item()


if __name__ == '__main__':

    X_novel = torch.from_numpy(np.load("../data/tmp/X_novel.npy").astype(np.float32))
    X_test = torch.from_numpy(np.load("../data/tmp/X_test.npy").astype(np.float32))
    table = np.load("../data/class_table.npy")
    model_0 = torch.load('../models/Ultra_trained_hope.pkl').cuda()
    model_1 = torch.load('../models/Sia_trained_80.pkl').cuda()
    model_2 = torch.load('../models/Sia_trained_65.pkl').cuda()

    for s in range(1127,1200):
        random.seed(s)
        k = 5
        #sample = [random.randint(0, 79) for _ in range(k)]
        sample = [random.randint(0, 499) for _ in range(k)]

        few_shot = np.zeros((k,20,3,32,32)).astype(np.float32)

        for i in range(len(X_novel)):
            for j in range(k):
                few_shot[j][i] = X_novel[i][sample[j]]
        
        few_shot = torch.from_numpy(few_shot)

        count = 0
        with torch.no_grad():
            model_0.eval()
            model_1.eval()
            model_2.eval()

            for label in range(20):
                for num in range(100):
                    predict = get_max(few_shot,X_novel[label][num],model_0,model_1,model_2)
                    if (predict == label):
                        count += 1
        
        if (count > 0):
            print ("Seeds: ",s)
            print ("Acc: ",float(count/2000))

        predict = list()

    for i in range(2000):
        if (i%100 == 0):
            print (i)
        x = X_test[i]
        tmp = get_max(few_shot,x,model_0,model_1,model_2)
        predict.append(table[tmp])

    f = open("YO.csv","w")
    f.write('image_id,predicted_label\n')

    for i in range(2000):
        f.write(str(i) + "," + str(predict[i])+"\n")

    f.close()        
