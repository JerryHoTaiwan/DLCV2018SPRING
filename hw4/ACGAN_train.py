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
from torchsummary import summary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(56)

num_epoch = 200
batch_size = 200
latent_dim = 100
is_train = 1

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

    train_label = torch.from_numpy(pd.read_csv('../hw4_data/train.csv')['Smiling'].as_matrix()).view(-1,1)
    test_label = torch.from_numpy(pd.read_csv('../hw4_data/test.csv')['Smiling'].as_matrix()).view(-1,1)

    x_train = np.load('../data/X_train.npy').astype(np.float32)
    x_test = np.load('../data/X_test.npy').astype(np.float32)

    x_train = np.rollaxis(x_train, 3, 1)  
    x_test = np.rollaxis(x_test, 3, 1)

    x_train = torch.from_numpy((x_train-127.5)/127.5)
    x_test = torch.from_numpy((x_test-127.5)/127.5)

    myTrain = data.TensorDataset(x_train, train_label)
    myTest = data.TensorDataset(x_test, test_label)

    dataloader = data.DataLoader(myTrain, batch_size=batch_size, shuffle=True, num_workers=4)
    loss_dis = nn.BCELoss()
    loss_class = nn.BCELoss()

    D = Discriminator().cuda()
    G = Generator().cuda()

    summary(D,(3,64,64))

    D_opt = optim.Adam(D.parameters(),lr=0.0001, betas=(0.5, 0.999))
    G_opt = optim.Adam(G.parameters(),lr=0.0001, betas=(0.5, 0.999))

    if(is_train):
        fixed_noise = torch.randn((16,latent_dim-1))
        fixed_noise = torch.cat((fixed_noise,fixed_noise),0)
        fixed_label = torch.cat((torch.zeros((16,1)),torch.ones((16,1))),0)
        fixed_noise = torch.cat((fixed_noise,fixed_label),1).cuda().view(-1,latent_dim,1,1)
        history = np.zeros((num_epoch,6))

        print ('starting training process...')
        for epoch in range(num_epoch):
          start = time.time()
          real_acc = 0.0
          fake_acc = 0.0
          real_attr_loss = 0.0
          fake_attr_loss = 0.0
          G_loss = 0.0
          D_loss = 0.0

          for i, (data,label) in enumerate(dataloader):
            
            #train D
            batch_size = data.size(0)
            output, attr = D(Variable(data).cuda())

            real_class = loss_class(attr, Variable(label.cuda()).float())
            real_dis = loss_dis(output, Variable(torch.ones(batch_size,1)).cuda())
            real_loss = real_dis + real_class
            real_acc_batch = np.mean(((output > 0.5).cpu().data.numpy()))

            # train with fake
            random_label = torch.from_numpy(np.random.randint(2, size=batch_size).reshape(batch_size, 1)).type(torch.FloatTensor)
            noise = torch.cat((torch.randn(batch_size,latent_dim-1),random_label),1).view(-1,latent_dim,1,1)
            
            fake = G(Variable(noise).cuda()).cuda()
            output, attr = D(fake.detach())
            
            fake_class = loss_class(attr, Variable(random_label.cuda()))
            fake_dis = loss_dis(output, Variable(torch.zeros(batch_size,1)).cuda())
            fake_loss = fake_dis + fake_class
            fake_acc_batch = np.mean(((output < 0.5).cpu().data.numpy()))

            D_batch_loss = real_loss + fake_loss
            D.zero_grad()
            real_loss.backward()
            fake_loss.backward()
            D_opt.step()

            #----------------------------------------------------
            # train G
            G.zero_grad()
            output, attr = D(fake)
            G_dis = loss_dis(output, Variable(torch.ones(batch_size,1).cuda()))
            G_class = loss_class(attr, Variable(random_label.cuda()))
            G_batch_loss = G_dis + G_class
            G_batch_loss.backward()
            gen_acc_batch = np.mean(((output > 0.5).cpu().data.numpy()))
            G_opt.step()
            
            real_acc += real_acc_batch
            fake_acc += fake_acc_batch
            real_attr_loss += real_class
            fake_attr_loss += fake_class
            G_loss += G_batch_loss
            D_loss += D_batch_loss

            if i == len(dataloader)-1:
              fake = G(Variable(fixed_noise))
              vutils.save_image(fake.detach().data,'../ACGAN_predict/fake_samples_epoch_%03d.png'% ( epoch),normalize=True)

            end = time.time()
            print('Epoch %d: [%d/%d] %d sec Dis_loss: %.4f Gen_loss: %.4f real: %.4f fake: %.4f gen: %.4f'
                  % (epoch+1, i+1  , len(dataloader),(end-start),
                    D_batch_loss, G_batch_loss, real_acc_batch, fake_acc_batch, gen_acc_batch), flush=True,end='\r')
                
          print ('')
          torch.save(G.state_dict(), '../checkpoint/AC_G_epoch_%d.pth' % (epoch))
          torch.save(D.state_dict(), '../checkpoint/AC_D_epoch_%d.pth' % (epoch))
          history[epoch] = [real_acc,fake_acc,G_loss,D_loss,real_attr_loss,fake_attr_loss]
          np.save('../checkpoint/ACGAN_hist_%d.npy'%(epoch),history)

          torch.save(G, '../models/ACGAN_G_%d.pkl'%(epoch))
          torch.save(D, '../models/ACGAN_D_%d.pkl'%(epoch))
