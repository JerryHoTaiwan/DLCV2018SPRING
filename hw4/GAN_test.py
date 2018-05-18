from sys import argv
import numpy as np
import os
import time
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(93)

EPOCH = 200
batch_size = 200
latent_dim = 100
is_test = 1
latent_size = 100

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
        return output
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(

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
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(4,4), stride=(1,1), padding=0, bias=False),
            nn.Sigmoid()
    )
        
        
    def forward(self, X):
        output = self.disc(X)

        return output.view(-1, 1).squeeze(1)

if __name__ == "__main__":

    if (is_test):
        hist = np.load('data/GAN_hist_199.npy')/214
        plt.figure(1)
        plt.subplots_adjust(wspace=None, hspace=0.5)
        plt.subplot(2,1,1)
        plt.plot(hist[:,2], label='real')
        plt.plot(hist[:,3], label='fake')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Discriminator')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(hist[:,1]*200, label='Gen_loss')
        plt.plot(hist[:,0]*200, label='Dis_loss')
        plt.title('Training Loss of Generator and Discriminator')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(argv[2], 'fig2_2.jpg'))

        model = torch.load('GAN_Generator.pkl').cuda()
        random_noise = torch.randn(32, latent_dim,1,1)

        random_img = model(Variable(random_noise.cuda()))
        vutils.save_image(random_img.detach().data,os.path.join(argv[2], 'fig2_3.jpg'),normalize=True)