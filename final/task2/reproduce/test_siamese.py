import random
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
from torch.utils.data import DataLoader
import torchvision

from modules import Siamese
from readfile import load_novel, load_test

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_max(few_shot,x,model):
    
    k = few_shot.size()[0]
    output = torch.zeros([20]).cuda()
    for i in range(k):
        xr = x.view(-1,3,32,32)
        xr = xr.repeat(few_shot.size()[1],1,1,1)

        f = few_shot[i].cuda()
        xr = xr.cuda()
        output += model(f,xr)
    return torch.argmax(output).item()


if __name__ == '__main__':

    novel_path = sys.argv[1]
    test_path = sys.argv[2]

    k = int(sys.argv[3])
    print (k,"-shots")

    X_novel, table = load_novel(novel_path)
    X_test = load_test(test_path).astype(np.float32)

    X_novel = torch.from_numpy(X_novel.astype(np.float32))
    X_test = torch.from_numpy(X_test)

    model = torch.load('Sia_'+str(k)+'shot.pkl').cuda()

    for s in range(7,8):
        random.seed(s)
        sample = [random.randint(0, 499) for _ in range(k)]

        few_shot = np.zeros((k,20,3,32,32)).astype(np.float32)

        for i in range(len(X_novel)):
            for j in range(k):
                few_shot[j][i] = X_novel[i][sample[j]]
        
        few_shot = torch.from_numpy(few_shot)
        predict = list()

    for i in range(2000):
        if (i%100 == 0):
            print (i)
        x = X_test[i]
        tmp = get_max(few_shot,x,model)
        predict.append(table[tmp])

    f = open("predict_"+str(k)+".csv","w")
    f.write('image_id,predicted_label\n')

    for i in range(2000):
        f.write(str(i) + "," + str(predict[i])+"\n")

    f.close()        
