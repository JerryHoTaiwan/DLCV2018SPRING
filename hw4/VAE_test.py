import numpy as np
import sys
import time
import os

from scipy.ndimage import imread
from scipy.misc import imsave
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# third-party library
import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from torch.optim import *
import torchvision
import torchvision.utils as vutils

#filepath = '../models/' + sys.argv[1]
#mode = int(sys.argv[2])

def read_faces(filepath):
	file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
	file_list.sort()
	n_faces = len(file_list)
	print (n_faces)
	faces = np.empty((n_faces, 64, 64, 3),dtype=np.uint8)

	for i, file in enumerate(file_list):
		if (i % 100 == 0):
			print (i,file)
		face = imread(os.path.join(filepath, file))
		faces[i] = face

	return faces

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

latent_dim = 512
BATCH_SIZE = 100
LR = 0.002
EPOCH = 100
patience = 10
Lambda = 0.00001
torch.manual_seed(888)

test_path = os.path.join(sys.argv[1]+'/test/')
X_valid = read_faces(test_path)

#X_train = np.load('../data/X_train.npy').astype(np.float32)

#X_train /= 255
X_test = (X_valid-127.5) / 127.5

#X_train = torch.from_numpy(X_train).type(torch.FloatTensor).view(-1,3,64,64)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).view(-1,3,64,64)

model = torch.load('VAE_97_model2.pkl').cuda()

if __name__ == "__main__":

	model.eval()
	pred = torch.FloatTensor()
	pred = pred.cuda()

	enc = torch.FloatTensor()
	enc = enc.cuda()

	std = torch.FloatTensor()
	std = std.cuda()

	# ======= training curve =======

	plt.figure(1)
	plt.subplots_adjust(wspace=None, hspace=0.5)
        
	plt.subplot(2,1,1)

	train_MSE = np.load("data/MSE2.npy")
	train_KLD = np.load("data/KLD2.npy")

	plt.plot(train_KLD[3:])
	plt.title("training KL divergence")
	plt.xlabel("steps")
	#plt.savefig("../VAE_predict/KL.png")

	plt.subplot(2,1,2)
	plt.plot(train_MSE)
	plt.title("training MSE")
	plt.xlabel("steps")
	plt.savefig(os.path.join(sys.argv[2], 'fig1_2.jpg'))
	plt.close()

	# =========== reconstruction ================
	
	pred = torch.FloatTensor()
	pred = pred.cuda()
	truth = torch.FloatTensor()
	truth = truth.cuda()
	
	for index in range(0,len(X_test) ,BATCH_SIZE):
		if index+BATCH_SIZE > len(X_valid):
			break
		input_X = X_test[index:index+BATCH_SIZE]
		input_X = Variable(input_X.cuda())
		X_pred,mu,logvar = model(input_X)
		
		pred = torch.cat((pred,X_pred.data),0)
		truth = torch.cat((truth,input_X.data),0)

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
		imsave(os.path.join(sys.argv[2], 'fig1_3.jpg'),save_com)

	# =========== random generation =============

	print ("random generation...")

	ran_noise = torch.randn((32,latent_dim))#.view(-1,latent_dim,1)
	test_img = model.decode(Variable(ran_noise.cuda()))
	#vutils.save_image(test_img.detach().data,os.path.join(sys.argv[2], 'fig1_4.jpg'),nrow=8, normalize=True)
	test_img = (test_img.data.cpu().numpy()+1) * 127.5

	tmp_1 = test_img[0].reshape(64,64,3)
	tmp_2 = test_img[8].reshape(64,64,3)
	tmp_3 = test_img[16].reshape(64,64,3)
	tmp_4 = test_img[24].reshape(64,64,3)

	for i in range(1,8):
		tmp_1 = np.concatenate((tmp_1,test_img[i].reshape(64,64,3)),axis=1)
		tmp_2 = np.concatenate((tmp_2,test_img[8+i].reshape(64,64,3)),axis=1)
		tmp_3 = np.concatenate((tmp_3,test_img[16+i].reshape(64,64,3)),axis=1)
		tmp_4 = np.concatenate((tmp_4,test_img[24+i].reshape(64,64,3)),axis=1)

	tmp_5 = np.concatenate((tmp_1,tmp_2),axis=0)
	tmp_6 = np.concatenate((tmp_3,tmp_4),axis=0)
	save_img = np.concatenate((tmp_5,tmp_6),axis=0)

	imsave(os.path.join(sys.argv[2], 'fig1_4.jpg'),save_img)

	# ======= TSNE =========

	test_attr = pd.read_csv(os.path.join(sys.argv[1], 'test.csv'))

	#for i in range(len(X_test)):
	input_X = Variable(X_test.view(-1,3,64,64).cuda())
	recon, mu, logvar = model(input_X)
	enc = torch.cat((enc,mu.data),0)
	std = torch.cat((std,logvar.data),0)

	pred = pred.cpu().numpy()[:10]
	enc = enc.cpu().numpy()
	std = std.cpu().numpy()

	print (np.mean(enc),np.mean(std))

	print ("TSNE...")

	latent_embedded = TSNE(n_components=2, perplexity=30.0, random_state=38).fit_transform(enc)
	
	attr_arr = np.array(test_attr["Male"])

	for i in [0,1]:
		if i == 1:
			color = "blue"
			gender = "Male"
		else:
			color = "red"
			gender = "Female"
		pos = latent_embedded[attr_arr==i]
		label = attr_arr[attr_arr==i]
		plt.scatter(pos[:,0], pos[:,1], c=color, label=gender,alpha=0.3)
	plt.title("Gender")
	plt.legend()
	plt.savefig(os.path.join(sys.argv[2], 'fig1_5.jpg'))
	plt.close()

	"""
	attr_arr = np.array(test_attr["Wearing_Lipstick"])
	
	for i in [0,1]:
		if i==1:
			color = "red"
			attr = "Lipstick"
		else:
			color = "blue"
			attr = "No Lipstick"
		xy = latent_embedded[attr_arr==i]
		label = attr_arr[attr_arr==i]
		plt.scatter(xy[:,0], xy[:,1], c=color, label=attr,alpha=0.3)
	plt.title("Lipstick")
	plt.savefig("../VAE_predict/TSNE_lipstick.png")
	plt.close()
	"""


