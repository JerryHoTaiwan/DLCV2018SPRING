import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

"""

total images for training: 500 * 80

"""

if __name__ == "__main__":

	path = "../task2-dataset/base/"
	folders = list()

	X_train = np.zeros((500*80*2,32,32,3)).astype(np.float32)
	Y_train = np.zeros((500*80*2,1))
	X_test = np.zeros((100*80,32,32,3)).astype(np.float32)
	Y_test = np.zeros((100*80,1))

	for folder in listdir(path):
		folders.append(folder)

	folders = sorted(folders)

	i = 0
	j = 0
	label = 0

	for folder in folders:
		train_path = path + folder + "/train/"
		test_path = path + folder + "/test/"

		for img in listdir(train_path):
			img_path = train_path + img
			X_train[2*i] = cv2.imread(img_path)
			X_train[2*i+1] = cv2.flip(cv2.imread(img_path),1)
			Y_train[2*i] = label
			Y_train[2*i+1] = label
			i += 1
		for img in listdir(test_path):
			img_path = test_path + img
			X_test[j] = cv2.imread(img_path)
			Y_test[j] = label
			j += 1
		label += 1

	X_train = (X_train - 127.5) / 127.5
	X_test = (X_test - 127.5) / 127.5

	np.save("../data/X_base_train.npy",X_train)
	np.save("../data/Y_base_train.npy",Y_train)
	np.save("../data/X_base_test.npy",X_test)
	np.save("../data/Y_base_test.npy",Y_test)

	print ("Finish loading!")
	
	path = "../task2-dataset/novel/"
	folders = list()
	classes = list()

	X_train = np.zeros((500*20,32,32,3)).astype(np.float32)
	Y_train = np.zeros((500*20,1))
	X_test = np.zeros((2000,32,32,3)).astype(np.float32)

	for folder in listdir(path):
		folders.append(folder)

	folders = sorted(folders)

	i = 0
	label = 0

	for folder in folders:
		train_path = path + folder + "/train/"
		classes.append(folder[6:8])
		for img in listdir(train_path):
			img_path = train_path + img
			X_train[i] = cv2.imread(img_path)
			Y_train[i] = label
			i += 1
		label += 1

	X_train = (X_train - 127.5) / 127.5

	np.save("../data/X_novel_train.npy",X_train)
	np.save("../data/Y_novel_train.npy",Y_train)

	path = "../test/"

	i = 0
	for img in listdir(path):
		img_path = path + img
		X_test[i] = cv2.imread(img_path)
		i += 1

	X_test = (X_test - 127.5) / 127.5

	np.save("../data/X_novel_test.npy",X_test)
	np.save("../data/class_table.npy",np.array(classes))