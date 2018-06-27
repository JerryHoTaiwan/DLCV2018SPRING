import numpy as np
import sys
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

	feature_dir = sys.argv[1]
	save_dir = sys.argv[2]
	train_num = int(sys.argv[3])
	valid_num = int(sys.argv[4])

	npy_train = list()
	npy_valid = list()

	for features in listdir(feature_dir):
		if (len(features) == 12):
			npy_train.append(features)
		elif (len(features) == 16):
			npy_valid.append(features)

	npy_train = sorted(npy_train)
	npy_valid = sorted(npy_valid)

	table = list()
	x_train = np.zeros((20*train_num,512))
	y_train = np.zeros(20*train_num)
	x_valid = np.zeros((20*valid_num,512))
	y_valid = np.zeros(20*valid_num)

	for i,item in enumerate(npy_train):
		print (i,item)
		FV = np.load(feature_dir + item)
		x_train[i*train_num:(i*train_num+train_num)] = FV
		y_train[i*train_num:(i*train_num+train_num)] = i
		table.append(item[6:8])

	for i,item in enumerate(npy_valid):
		print (i,item)
		FV = np.load(feature_dir + item)
		x_valid[i*valid_num:(i*valid_num+valid_num)] = FV
		y_valid[i*valid_num:(i*valid_num+valid_num)] = i

	print (y_train[-1],y_valid[-1])

	np.save(join(save_dir,'x_feature_train.npy'),x_train)
	np.save(join(save_dir,'y_feature_train.npy'),y_train)
	np.save(join(save_dir,'x_feature_valid.npy'),x_valid)
	np.save(join(save_dir,'y_feature_valid.npy'),y_valid)
