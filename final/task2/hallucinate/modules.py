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

class feature2classes(nn.Module):
    def __init__(self,feature_size=512,num_classes=20):
        super(feature2classes, self).__init__()

        self.fc1 = nn.Linear(feature_size,128)
        self.fc2 = nn.Linear(128,num_classes)
        self.bn1 = nn.BatchNorm1d(128)
        self.dp = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                size = m.weight.size()
                fan_in = size[0]
                fan_out = size[1]
                n = fan_in + fan_out
                m.weight.data.normal_(0,math.sqrt(2./n))
            if isinstance(m,nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,fv):
        out = self.fc1(fv)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dp(out)
        out = self.fc2(out)
        return out

class base_classifier(nn.Module):
    def __init__(self):
        super(base_classifier, self).__init__()
        self.img2fv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
		    nn.BatchNorm2d(64),
            nn.RReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(128),
            nn.RReLU(),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(256),
            nn.RReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            #nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(512),
            nn.RReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4)       	
            )
        self.fv2label = nn.Sequential(
            nn.Linear(8192,512),
		    nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,80),
		    nn.BatchNorm1d(80),
            nn.Softmax(1)
            )

        for m in self.modules():
        	if isinstance(m,nn.Conv2d):
        		n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
        		m.weight.data.normal_(0,math.sqrt(2./n))
        	if isinstance(m,nn.BatchNorm2d):
        		m.weight.data.fill_(1)
        		m.bias.data.zero_()

    def forward(self,x):
        #print (x)
        fv = self.img2fv(x)
        fv = fv.view(fv.size(0),-1)
        output = self.fv2label(fv)
        return fv,output

class cheat_classifier(nn.Module):
    def __init__(self):
        super(cheat_classifier, self).__init__()
        self.img2fv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
		    nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            #nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
		    nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4)       	
            )
        self.fv2label_cheat = nn.Sequential(
            nn.Linear(8192,512),
		    nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,20),
		    nn.BatchNorm1d(20),
            nn.Softmax(1)
            )

        for m in self.modules():
        	if isinstance(m,nn.Conv2d):
        		n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
        		m.weight.data.normal_(0,math.sqrt(2./n))
        	if isinstance(m,nn.BatchNorm2d):
        		m.weight.data.fill_(1)
        		m.bias.data.zero_()

    def forward(self,x):
        #print (x)
        fv = self.img2fv(x)
        fv = fv.view(fv.size(0),-1)
        output = self.fv2label_cheat(fv)
        return fv,output
