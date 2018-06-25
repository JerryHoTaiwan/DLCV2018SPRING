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

if __name__ == '__main__':

	fv1 = np.load("../data/X_train_fv.npy")
	fv2 = np.load("../data/X_train_fv2.npy")
	fv3 = np.load("../data/X_train_fv3.npy")
	fv4 = np.load("../data/X_train_fv4.npy")

	fv = (fv1 + fv2 + fv3 + fv4)
	np.save("../data/X_train_fv_all.npy",fv)