import os
import sys
import csv
import collections
import numpy as np
import pandas as pd
import time
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F

import skvideo.io
import skimage.transform
from scipy.ndimage import imread
from scipy.misc import imsave

import pickle

feature_size = 512 * 7 * 7
rescale_factor = 1
feature_size = 25088

class Seq2Seq_Classifier(nn.Module):

    def __init__(self):
        super(Seq2Seq_Classifier, self).__init__()
        self.gru = nn.GRU(512*7*7, 512, num_layers=2, dropout=0.5,bidirectional=True,batch_first=True)

        self.classifier = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 11)
        )

    def forward(self, x):
       # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        len = x.size()[0]
        x = x.view(-1,len,feature_size)
        out, hn = self.gru(x, None)
        y = self.classifier(out.squeeze())
        y = F.softmax(y, 1)
        #y = F.relu(self.fc2(y))
        return y

def read_img(filepath):
	file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
	file_list.sort()
	n_img = len(file_list)
	imgs = np.empty((n_img, 224, 224, 3),dtype=np.float32)

	for i, file in enumerate(file_list):
		img = imread(os.path.join(filepath, file))
		img = skimage.transform.rescale(img, rescale_factor, mode='constant', preserve_range=True)
		img = skimage.transform.resize(img,(224,224))
		imgs[i] = img

	return imgs

if __name__ == '__main__':

	# ====== input path ==========

	video_path = sys.argv[1]
	output_folder = sys.argv[2]

	# ====== loading data ========

	print('loading pretrained model...')
	model = torchvision.models.vgg16(pretrained=True).features.cuda()

	print('reading data...')

	x = []
	y = []
	cat_all = []

	for category in os.listdir(video_path):
		print (category)
		cat_all.append(category)
		cat_path = video_path + category + "/"
		data_x = read_img(cat_path)
		data_x = np.rollaxis(data_x ,3 ,1)
		data_x = (data_x.astype(np.float32)-127.5)/127.5

		iteration = int(data_x.shape[0]/60)

		if (data_x.shape[0] > 60):
			output = model(torch.from_numpy(data_x[0:60,:,:,:]).cuda()).detach().cpu().reshape(-1,512*7*7)
			for i in range(1,iteration):
				output_tmp = model(torch.from_numpy(data_x[(i*60):(i*60+60),:,:,:]).cuda()).detach().cpu().reshape(-1,512*7*7)
				output = torch.cat((output,output_tmp),0)

			output_tmp = model(torch.from_numpy(data_x[(iteration*60):,:,:,:]).cuda()).detach().cpu().reshape(-1,512*7*7)
			output = torch.cat((output,output_tmp),0)
		else:
			output = model(torch.from_numpy(data_x).cuda()).detach().cpu().reshape(-1,512*7*7)

		#y_path = labelpath + category + ".txt"
		#label = np.array(pd.read_csv(y_path,header=None)).astype(np.long)

		x.append(output)
		#y.append(torch.from_numpy(label))

	print (cat_all)

	x_valid = x
#train = read_img(filepath + )

	model = torch.load('seq2seq_5920.pkt').cuda()
	model.eval()
	output_label = list()
	BATCH_SIZE = 64

	with torch.no_grad():
		val_loss = 0.0
		val_acc = 0.0
		for i in range(len(x_valid)):
			x_data = x_valid[i]

			index = 0
			input_X = x_data[index:index + BATCH_SIZE]
			predict = model(input_X.cuda())
			output = torch.argmax(predict, 1).cpu()
			
			#y_data = y_valid[i].type(torch.LongTensor)               
			for index in range(BATCH_SIZE,len(x_valid[i]),BATCH_SIZE):
				if (index + BATCH_SIZE > len(x_valid[i])):
					input_X = x_data[index:]
					#input_y = y_data[index:].squeeze()
				else:
					input_X = x_data[index:index + BATCH_SIZE]
					#input_y = y_data[index:index + BATCH_SIZE].squeeze()

				predict_tmp = model(input_X.cuda())
				output_tmp = torch.argmax(predict_tmp, 1).cpu()
				#val_loss += loss_function(output, input_y.cuda()).item()
				#val_acc += torch.sum((torch.argmax(output, 1).cpu() == input_y)).item()
				output = torch.cat((output,output_tmp),0)
			#print (len(output))
			output_label.append(output)
	#print ("Val_loss: ",val_loss, "Val_acc: ", float(val_acc/valid_length))

	# ======== Writing txt ===========

	for i in range(len(cat_all)):
		category = cat_all[i]
		txt_name = category + '.txt'
		f = open(os.path.join(output_folder, txt_name),'w')

		for j, pred in enumerate(output_label[i]):
			f.write(str(pred.item()))
			#if j != len(output_label[i])-1:
			f.write("\n")
		f.close()