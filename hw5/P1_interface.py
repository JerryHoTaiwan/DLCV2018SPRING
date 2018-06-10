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

import skvideo.io
import skimage.transform

class Net(torch.nn.Module):
	def __init__(self, feature_size):
		super(Net, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(feature_size, 4096),
			nn.BatchNorm1d(4096,momentum=0.5),
			nn.ReLU(),
			nn.Linear(4096, 1024),
			nn.BatchNorm1d(1024,momentum=0.5),
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.BatchNorm1d(256,momentum=0.5),
			nn.ReLU(),
			nn.Linear(256,11),
			nn.BatchNorm1d(11,momentum=0.5),
			nn.Softmax(1)
		)

	def forward(self, input):
		output = self.main(input)
		return output


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

			output = torch.mean(output,0)
			#output = model(torch.from_numpy(data_x).cuda()).detach().cpu().view(-1,512*7*7)
			#print (output)
			output = output.numpy().astype(np.float32)
			x.append(output)
			#x.append(np.mean(output, axis = 0))
			y.append(table['Action_labels'][i])

	valid_X = torch.from_numpy(np.array(x).astype(np.float32))
	valid_y = torch.from_numpy(np.array(y).astype(np.long)).view(-1)

	# ======== Testing ======== #

	CNN_model = torch.load('CNN_best.pkt').cuda()
	CNN_model.eval()
	loss_function = nn.CrossEntropyLoss()

	val_loss = 0.0
	val_acc = 0.0

	with torch.no_grad():
		model.eval()
		output = CNN_model(valid_X.cuda())
		val_loss = loss_function(output,valid_y.cuda()).item()
		output_label = torch.argmax(output,1).cpu().data
		val_acc = np.mean((output_label == valid_y).numpy())

	print ("Loss: ",val_loss, "Acc: ", val_acc)

	# ======== Writing txt ===========

	f = open(os.path.join(output_folder, 'p1_valid.txt'),'w')

	for i, pred in enumerate(output_label):
		f.write(str(pred.item()))
		if i != len(output_label)-1:
			f.write("\n")
	f.close()