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

class base_classifier(nn.Module):
    def __init__(self):
        super(base_classifier, self).__init__()
        self.img2fv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),stride=(1,1),padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
        	
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),       	
        	)
        self.fv2label = nn.Sequential(
			nn.Linear(4096,512),
			nn.ReLU(),
			#nn.Dropout(0.5),
			nn.Linear(512,80),
			nn.Softmax(1),
			)

        def forward(self,x):
        	fv = self.img2fv(x)
        	output = self.fv2label(fv)
        	return output
