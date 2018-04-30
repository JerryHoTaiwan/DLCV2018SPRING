import numpy as np
import sys
import os
from keras.models import Sequential,Model,load_model
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import  Adam,Nadam,Adamax
from keras.layers import *
from keras.callbacks import ModelCheckpoint,EarlyStopping
import h5py
from keras.regularizers import l2
from keras.utils import np_utils
from scipy.misc import *
import argparse
import os

color_label = np.array([[255,0,255],[255,255,0],[255,255,255],[0,0,0],[0,0,255],[0,255,0],[0,255,255]])
path = 'fcn_32.h5'
model = load_model(path)

def four_dig(num):
	if (num > 999):
		return str(num)
	elif (num > 99):
		return "0"+str(num)
	elif (num > 9):
		return "00"+str(num)
	elif (num > 0):
		return "000"+str(num)
	else:
		return "0000"

def read_images(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512,3))

    for i, file in enumerate(file_list):
        mask = imread(os.path.join(filepath, file))
        masks[i] = mask

    return masks

def pred2rgb(stack):
	rgb = np.zeros((len(stack)*512*512,3),dtype=np.uint8)
	tmp = stack.reshape(len(stack)*512*512,7)
	am = np.argmax(tmp,axis=1).astype(np.uint8)
	for i in range(len(am)):
		rgb[i] = color_label[am[i]]

	RGB = rgb.reshape(len(stack),512,512,3).astype(np.uint8)
	return RGB

if __name__ == '__main__':

	test_dir = sys.argv[1]
	output_dir = sys.argv[2]

	print ('loading...')
	X_test = read_images(test_dir).astype(np.float32)
	X_test /= 255

	print ('predicting...')
	result = model.predict(X_test,batch_size=10)

	print ('turning result 2 RGB...')
	RGB = pred2rgb(result)

	print (RGB.shape)

	for i in range(len(RGB)):
		path = output_dir + '/' + four_dig(i) + '_mask.png'
		imsave(path,RGB[i])