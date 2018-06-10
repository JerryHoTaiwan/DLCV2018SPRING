import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os import listdir
import os
import pandas as pd
import numpy as np
import sys
import csv
import time
import collections

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

import skvideo.io
import skimage.transform

class RNN_Classifier(nn.Module):

	def __init__(self):
		super(RNN_Classifier, self).__init__()
		self.gru = nn.GRU(512*7*7, 512, num_layers=2, dropout=0.5,bidirectional=True)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc1 = nn.Linear(512, 11)

	def forward(self, x, lengths):
		x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
		out, hn = self.gru(x, None)
		#print (x.data.size(),out.data.size(),hn.size())
		y = self.bn1(hn[-1])
		y = F.softmax(self.fc1(y), 1)
		#y = F.relu(self.fc2(y))
		return y,hn

def createBatch(x, y):
	sort_index = sorted(range(len(x)),key=lambda idx:len(x[idx]), reverse = True)
	x.sort(key=len, reverse=True)
	X = torch.nn.utils.rnn.pad_sequence(x)
	Y = torch.tensor([int(y[i]) for i in sort_index ])
	record_lengths = torch.tensor([len(x_) for x_ in x])

	return X, Y, record_lengths

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):

	filepath = video_path + '/' + video_category
	filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
	video = os.path.join(filepath,filename[0])

	videogen = skvideo.io.vreader(video)
	frames = []
	for frameIdx, frame in enumerate(videogen):
		if frameIdx % downsample_factor == 0:
			frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True)
			frame = skimage.transform.resize(frame,(224,224))
			frames.append(frame)
		else:
			continue

	return np.array(frames).astype(np.uint8)

def getVideoList(data_path):

	result = {}

	with open (data_path) as f:
		reader = csv.DictReader(f)
		for row in reader:
			for column, value in row.items():
				result.setdefault(column,[]).append(value)

	od = collections.OrderedDict(sorted(result.items()))
	return od

if __name__ == '__main__':

	# ====== input path ==========

	video_path = sys.argv[1]
	csv_path = sys.argv[2]
	output_folder = sys.argv[3]

	# ====== loading data ========

	print('loading pretrained model...')
	model = torchvision.models.vgg16(pretrained=True).features.cuda()

	print('reading data...')
	
	table = getVideoList(csv_path)
	x, y = [], []

	with torch.no_grad():
		for i in range(len(table['Video_category'])):
			print (i, end='\r', flush=True)
			data_x = readShortVideo(video_path+'/', table['Video_category'][i], table['Video_name'][i])
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

			#output = model(torch.from_numpy(data_x).cuda()).detach().cpu().view(-1,512*7*7)
			#print (output)
			#output = output.numpy().astype(np.float32)
			x.append(output)
			#x.append(np.mean(output, axis = 0))
			y.append(table['Action_labels'][i])

	x_val = x
	y_val = torch.from_numpy(np.array(y).astype(np.long)).view(-1)

	# ======== Testing ======== #


	RNN_model = torch.load('RNN_best.pkt').cuda()
	RNN_model.eval()
	CELoss = nn.CrossEntropyLoss()

	val_loss = 0.0
	val_acc = 0.0
	BATCH_SIZE = 1

	with torch.no_grad():

		i = 0
		input_X, input_Y, lengths_batch = createBatch(x_val[i:i+BATCH_SIZE], y_val[i:i+BATCH_SIZE])
		predict,_hn = RNN_model(input_X.cuda(), lengths_batch)

		batch_loss = CELoss(predict, input_Y.cuda())
		val_loss += batch_loss.item()
		val_acc += torch.sum((torch.argmax(predict, 1).cpu() == input_Y)).item()
		output_label = torch.argmax(predict, 1).cpu()

		for i in range(BATCH_SIZE,len(x_val),BATCH_SIZE):
			if ( i + BATCH_SIZE > len(x_val)):
				input_X, input_Y, lengths_batch = createBatch(x_val[i:],y_val[i:])
			else:
				input_X, input_Y, lengths_batch = createBatch(x_val[i:i+BATCH_SIZE], y_val[i:i+BATCH_SIZE])

			predict_tmp,_hn = RNN_model(input_X.cuda(), lengths_batch)

			batch_loss = CELoss(predict_tmp, input_Y.cuda())
			val_loss += batch_loss.item()
			val_acc += torch.sum((torch.argmax(predict_tmp, 1).cpu() == input_Y)).item()
			#print (torch.argmax(predict_tmp, 1).cpu())
			label_tmp = torch.argmax(predict_tmp, 1).cpu()
			output_label = torch.cat((output_label,label_tmp),0)

		print (len(output_label))
		print ("Val_loss: ",val_loss, "Val_acc: ", val_acc/len(x_val))

	# ======== Writing txt ===========

	f = open(os.path.join(output_folder, 'p2_result.txt'),'w')

	for i, pred in enumerate(output_label):
		f.write(str(pred.item()))
		if i != len(output_label)-1:
			f.write("\n")

	f.close()