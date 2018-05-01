import numpy as np
from scipy.ndimage import imread
#from skimage import transform
#from keras.utils import np_utils
import psutil

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

def img2cat(img,label):
	img = img.reshape(size*size,3)
	for i in range(len(img)):
		if img[i][0] == 255:
			if img[i][1] == 0:
				cat = 0
			else:
				if img[i][2] == 0:
					cat = 1
				else:
					cat = 2

		else:
			if img[i][1] == 0:
				if img[i][2] == 0:
					cat = 3
				else:
					cat = 4
			else:
				if img[i][2] == 0:
					cat = 5
				else:
					cat = 6
		label[i][cat] = 1


classes = 7
size = 512

train_path = 'hw3-train-validation/train/'
valid_path = 'hw3-train-validation/validation/'

if __name__ == '__main__':

	train = np.zeros((2313,size,size,3)).astype(np.float32)
	y_train0 = np.zeros((2313,size*size,3)).astype(np.uint8)
	y_train1 = np.zeros((2313,size*size,4)).astype(np.uint8)
	y_train = np.concatenate((y_train0,y_train1),axis=2)
	del y_train0
	del y_train1

	print (psutil.virtual_memory())

	valid = np.zeros((257,size,size,3)).astype(np.float32)
	y_valid = np.zeros((257,size*size,7)).astype(np.uint8)

	get_train = 1
	get_valid = 0
	if (get_train):
		for i in range(2313):
			print (i)
			name = imread(train_path + four_dig(i) + "_sat.jpg")
			label = imread(train_path + four_dig(i) + "_mask.png")
			train[i] = name #transform.resize(name,(256,256,3))
			img2cat(label,y_train[i])

	if (get_valid):
		for i in range(257):
			print (i)
			name = imread(valid_path + four_dig(i) + "_sat.jpg")
			label = imread(valid_path + four_dig(i) + "_mask.png")
			valid[i] = name #transform.resize(name,(256,256,3))
			img2cat(label,y_valid[i])

	print (psutil.virtual_memory())

	"""
	valid = valid.astype(np.uint8)
	valid_label = valid_label.astype(np.uint8)

	"""
	#np.save('data/train_256.npy',train)
	#y_train = np_utils.to_categorical(train_label, num_classes=classes).astype(np.uint8)
	#np.save('data/y_train_256.npy',y_train)

	#np.save('data/valid_256.npy',valid)
	#y_valid = np_utils.to_categorical(valid_label, num_classes=classes).astype(np.uint8)
	#np.save('data/y_valid_256.npy',y_valid)