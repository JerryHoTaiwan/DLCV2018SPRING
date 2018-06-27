import numpy as np
import sys
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

	feature_dir = sys.argv[1]
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
	x_train = np.zeros((20*3801,512))
	y_train = np.zeros(20*3801)
	x_valid = np.zeros((20*400,512))
	y_valid = np.zeros(20*400)

	for i,item in enumerate(npy_train):
		print (i,item)
		FV = np.load(feature_dir + item)
		x_train[i*3801:(i*3801+3801)] = FV
		y_train[i*3801:(i*3801+3801)] = i
		table.append(item[6:8])

	for i,item in enumerate(npy_valid):
		print (i,item)
		FV = np.load(feature_dir + item)
		x_valid[i*400:(i*400+400)] = FV
		y_valid[i*400:(i*400+400)] = i

	print (y_train[-1],y_valid[-1])

	np.save('../data/x_feature_train.npy',x_train)
	np.save('../data/y_feature_train.npy',y_train)
	np.save('../data/x_feature_valid.npy',x_valid)
	np.save('../data/y_feature_valid.npy',y_valid)
