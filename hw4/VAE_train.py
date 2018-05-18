import numpy as np
import matplotlib.pyplot as plt

import sys
import psutil
import time
import GPUtil

from util import *

# third-party library
import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.optim import *
from torchsummary import summary
import torchvision

from scipy.misc import imsave

#filepath = '../models/' + sys.argv[1]
#mode = int(sys.argv[2])

BATCH_SIZE = 100
LR = 0.002
EPOCH = 100
patience = 30
Lambda = 5

X_train = np.load('../data/X_train.npy').astype(np.float32)
X_valid = np.load('../data/X_test.npy').astype(np.float32)

X_train = (X_train-127.5)/127.5
X_valid = (X_valid-127.5)/127.5

#X_train /= 255
#X_valid /= 255

X_train = torch.from_numpy(X_train).type(torch.FloatTensor).view(-1,3,64,64)
X_valid = torch.from_numpy(X_valid).type(torch.FloatTensor).view(-1,3,64,64)

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.latent_dim = 512

		self.conv = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.01, inplace=True),
            
			nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.01, inplace=True),
            
			nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.01, inplace=True),
		)

		self.z_mean = nn.Linear(8192, self.latent_dim)
		self.z_std = nn.Linear(8192, self.latent_dim)
		self.z_dec = nn.Linear(self.latent_dim, 8192)
		self.tanh = nn.Tanh()

		self.transp = nn.Sequential(

			nn.ConvTranspose2d(512, 256, 4, 2, padding=1,bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.01, inplace=True),

			nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.01, inplace=True),
 
			nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.01, inplace=True),
            
			nn.ConvTranspose2d(64, 3, 4, 2, padding=1,bias=False)
		)
	def encode(self, img):
		enc = self.conv(img).view(-1, 8192)
		return self.z_mean(enc), self.z_std(enc)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps).cuda()

		return eps.mul(std).add_(mu)

	def decode(self, z):
		dec = self.z_dec(z).view(-1, 512, 4, 4)
		pred = self.transp(dec)
		return self.tanh(pred)#/2.0+0.5

	def forward(self, x):
		mu, logvar = self.encode(x)
		dis = self.reparameterize(mu, logvar)
		return self.decode(dis), mu, logvar

def loss_func(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    MSE = functional.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    VAE_loss = MSE + Lambda * KLD
    return MSE,KLD,VAE_loss

if __name__ == "__main__":

	model = VAE().cuda()

	optimizer = torch.optim.Adamax(model.parameters(), lr=LR)
	
	total_length = len(X_train)
	max_steps = int(total_length / BATCH_SIZE)

	Max_loss = 10000.0

	### training ###
	process_bar = ShowProcess(max_steps)

	train_MSE = []
	train_KLD = []
	test_record = []

	summary(model,(3,64,64))
	count = 0

	for epoch in range(EPOCH):
		print("Epoch:", epoch+1)
		tstart = time.time()

		MSE_loss = 0.0
		KL_loss = 0.0
		valid_mse = 0.0

		# shuffle
		perm_index = torch.randperm(total_length)
		train_X_sfl = X_train[perm_index]

		# construct training batch
		for index in range(0,total_length ,BATCH_SIZE):
			if index+BATCH_SIZE > total_length:
				break

			process_bar.show_process()
			# zero the parameter gradients
			optimizer.zero_grad()
			input_X = train_X_sfl[index:index+BATCH_SIZE]

		    # use GPU
			input_X = Variable(input_X.cuda())

			# forward + backward + optimize
			
			X_pred,mu,logvar = model(input_X)
			MSE,KLD,vae_loss = loss_func(X_pred, input_X, mu, logvar)			
			vae_loss.backward()
			optimizer.step()
			MSE_loss += MSE.data[0]
			KL_loss += KLD.data[0]
			train_MSE.append(MSE.data[0]/(12288*BATCH_SIZE))
			train_KLD.append(KLD.data[0]/BATCH_SIZE)

		process_bar.close()
		print("Pixelwise MSE:",MSE_loss/(12288*total_length))
		print("KL divergence:",KL_loss/(total_length))

		# validation evaluation
		
		model.eval()

		pred = torch.FloatTensor()
		pred = pred.cuda()
		truth = torch.FloatTensor()
		truth = truth.cuda()

		for index in range(0,len(X_valid) ,BATCH_SIZE):
			if index+BATCH_SIZE > len(X_valid):
				break
			input_X = X_valid[index:index+BATCH_SIZE]
			input_X = Variable(input_X.cuda())
			X_pred,mu,logvar = model(input_X)
			
			pred = torch.cat((pred,X_pred.data),0)
			truth = torch.cat((truth,input_X.data),0)

			#loss = loss_func(output,input_X)
			test_mse = functional.mse_loss(X_pred, input_X,size_average=False)
			valid_mse += test_mse.data[0]
			test_record.append(test_mse.data[0]/BATCH_SIZE)

		if (valid_mse/(len(X_valid)) < Max_loss):
			Max_loss = valid_mse/(len(X_valid))
			print ("The valid MSE is improved to : ",valid_mse/(12288*len(X_valid))," Ｕ^皿^Ｕ")
			print ("saving model...")
			torch.save(model, '../models/VAE_e05.pkl')
			count = 0
		else:
			print("Validation Loss: ",valid_mse/(12288*len(X_valid)), "doesn't improve (┛◉Д◉)┛彡┻━┻")
			count += 1

		if (True):
			print ("saving image...")

			#print (Variable(pred).data)
			pred_n = Variable(pred).cpu().data.numpy()[:10]
			truth_n = Variable(truth).cpu().data.numpy()[:10]

			#truth = np.copy(X_valid[:10].numpy())
			
			pred_n = (pred_n+1) * 127.5
			truth_n = (truth_n+1) * 127.5

			#pred_n *= 255
			#truth_n *= 255

			pred_tmp = pred_n[0].reshape(64,64,3)
			truth_tmp = truth_n[0].reshape(64,64,3)

			for i in range(1,10):
				pred_tmp = np.concatenate((pred_tmp,pred_n[i].reshape(64,64,3)),axis=1)
				truth_tmp = np.concatenate((truth_tmp,truth_n[i].reshape(64,64,3)),axis=1)

				#imsave("../predict/truth_"+str(epoch)+"_"+str(i)+".png",truth[i].reshape(64,64,3).astype(np.uint8))
				#imsave("../VAE_predict/recon5_"+str(epoch)+"_"+str(i)+".png",pred_n[i].reshape(64,64,3).astype(np.uint8))
			save_com = np.concatenate((pred_tmp,truth_tmp),axis=0)
			imsave("../VAE_predict/compare_%d.png"%(epoch),save_com)

		model.train()
		tend = time.time()

		print("Time: ",tend - tstart)

		if (count == patience):
			break

		torch.save(model,'../models/VAE_%d_model2.pkl'%(epoch))
		torch.save(model.state_dict(),'../models/VAE_%d_model2.pth'%(epoch))

	train_MSE = np.array(train_MSE)
	train_KLD = np.array(train_KLD)
	test_record = np.array(test_record)

	np.save("../checkpoint/MSE2.npy",train_MSE)
	np.save("../checkpoint/KLD2.npy",train_KLD)