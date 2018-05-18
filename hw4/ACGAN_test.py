from sys import argv
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils import data
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(499)

num_epoch = 200
batch_size = 200
latent_dim = 100
is_test = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(

            nn.ConvTranspose2d( latent_dim, 512, 4, 1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),

            nn.ConvTranspose2d(512, 256,4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),

            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.4),

            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, padding=1,  bias=False),
            nn.Tanh()
        )
        
        
    def forward(self, X):
        output = self.gen(X)#/2.0+0.5
        return output.squeeze()
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encode = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256,kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512,kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

    )
        self.dis = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.classification = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, X):
        output = self.encode(X)
        out_dis = nn.Sigmoid()(self.dis(output))
        out_class = nn.Sigmoid()(self.classification(output))

        return (out_dis.view(-1, 1).squeeze(),out_class.view(-1, 1).squeeze())

if __name__ == "__main__":

    D = Discriminator().cuda()
    G = Generator().cuda()

    if (is_test):
        hist = np.load('data/ACGAN_hist_50.npy')/214
        plt.figure(1)
        plt.subplots_adjust(wspace=None, hspace=0.5)
        
        plt.subplot(2,1,1)
        plt.plot(hist[:,4], label='real')
        plt.plot(hist[:,5], label='fake')
        plt.title('Training Loss of Attribute Classification')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(hist[:,0], label='real')
        plt.plot(hist[:,1], label='fake')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Discriminator')
        plt.legend()
        
        plt.savefig(os.path.join(argv[2], 'fig3_2.jpg'))
        plt.close()

        model = torch.load('ACGAN_G_199.pkl').cuda()
        
        test_noise = torch.randn((10,latent_dim-1))
        dup_noise = torch.cat((test_noise,test_noise),0)
        compare = torch.cat((torch.zeros((10,1)),torch.ones((10,1))),0)
        comb_noise = torch.cat((dup_noise,compare),1).view(-1,latent_dim,1,1)
        test_img = model(Variable(comb_noise.cuda()))
        vutils.save_image(test_img.detach().data,os.path.join(argv[2], 'fig3_3.jpg'),nrow=10, normalize=True)
