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

torch.manual_seed(666)

EPOCH = 200
batch_size = 200
latent_dim = 100
train = 1
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

    x_train = np.load('../data/X_train.npy').astype(np.float32)
    x_test = np.load('../data/X_test.npy').astype(np.float32)
    x_train = np.concatenate((x_train,x_test),axis=0)
    x_train = np.rollaxis(x_train, 3, 1)  
    x_train = (x_train-127.5)/127.5

    dataloader = data.DataLoader(x_train, batch_size=batch_size, shuffle=True, num_workers=4)
    loss = nn.BCELoss()

    D = Discriminator().cuda()
    G = Generator().cuda()

    D_opt = optim.Adam(D.parameters(),lr=0.0001, betas=(0.5, 0.999))
    G_opt = optim.Adam(G.parameters(),lr=0.0001, betas=(0.5, 0.999))

    fixed_noise = torch.randn((32,latent_dim,1,1)).cuda()
    record = np.zeros((EPOCH,4))

    if(train):

        print ('training...')
        for epoch in range(EPOCH):
          start = time.time()
          G_loss = 0.0
          D_loss = 0.0
          real_acc = 0.0
          fake_acc = 0.0
          for i, data in enumerate(dataloader):

            batch_size = data.size(0)
            output = D(Variable(data.cuda()))
            real_loss = loss(output, Variable(torch.ones(batch_size)).cuda()) 
            real_acc_batch = np.mean(((output > 0.5).cpu().data.numpy()))

            fake_vec = torch.randn(batch_size,latent_dim,1,1)
            fake_img = G(Variable(fake_vec.cuda()))
            output = D(fake_img.detach()).squeeze()
            fake_loss = loss(output, Variable(torch.zeros(batch_size)).cuda())
            fake_acc_batch = np.mean(((output < 0.5).cpu().data.numpy()))

            D_batch_loss = real_loss + fake_loss

            D.zero_grad()
            D_batch_loss.backward()
            D_opt.step()

            # ========================================
            output = D(fake_img)
            G_batch_loss = loss(output, Variable(torch.ones(batch_size)).cuda())
            gen_acc_batch = np.mean(((output > 0.5).cpu().data.numpy()))

            G.zero_grad()
            G_batch_loss.backward()
            G_opt.step()
            
            real_acc += real_acc_batch
            fake_acc += fake_acc_batch
            G_loss += G_batch_loss
            D_loss += D_batch_loss

            if (i+1 == len(dataloader)):
              fake = G(Variable(fixed_noise.cuda()))
              vutils.save_image(fake.detach().data,'../GAN_predict/samples_epoch_%03d.png'% ( epoch),normalize=True)

            end = time.time()
            print('Epoch %d: [%d/%d] %d sec Dis_loss: %.4f Gen_loss: %.4f real: %.4f fake: %.4f gen: %.4f'
                  % (epoch+1, i+1  , len(dataloader),(end-start),
                    D_batch_loss, G_batch_loss, real_acc_batch, fake_acc_batch, gen_acc_batch), flush=True,end='\r')
                
          print ('')
          torch.save(G.state_dict(), '../checkpoint/G_epoch_%d.pth' % (epoch))
          torch.save(D.state_dict(), '../checkpoint/D_epoch_%d.pth' % (epoch))
          record[epoch] = [D_batch_loss,G_batch_loss,real_acc,fake_acc]
          np.save('../checkpoint/GAN_hist_%d.npy'%(epoch),record)

        torch.save(G, '../models/GAN_Generator.pkl')
        torch.save(D, '../models/GAN_Discriminator.pkl')