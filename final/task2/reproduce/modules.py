import sys
import numpy as np
import os
import math
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as Data
import torchvision 

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.img2fv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(1,1),stride=(1,1),padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.Sigmoid()
            )

        self.dis2sim = nn.Sequential(
            nn.Linear(2048,128),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
            )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            if isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x1,x2):
        fv1 = self.img2fv(x1).view(-1,2048)
        fv2 = self.img2fv(x2).view(-1,2048)
        dis = torch.abs(fv1 - fv2)
        output = self.dis2sim(dis).squeeze()
        return output

class Siamese_Ultra(nn.Module):
    def __init__(self):
        super(Siamese_Ultra, self).__init__()
        self.img2fv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(1,1),stride=(1,1),padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.Sigmoid()
            )

        self.fv2dis = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(2,2),stride=(1,1),padding=(0,0))

        self.dis2score = nn.Sequential(
            nn.Linear(2047,128),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
            )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            if isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
                own_state[name].copy_(param)

    def forward(self,x1,x2):
        fv1 = self.img2fv(x1).view(-1,1,2048)
        fv2 = self.img2fv(x2).view(-1,1,2048)
        fv = torch.cat((fv1,fv2),1).view(-1,1,2,2048)
        dis = self.fv2dis(fv)#.squeeze()
        dis = dis.view(-1,2047)
        output = self.dis2score(dis).squeeze()
        return output