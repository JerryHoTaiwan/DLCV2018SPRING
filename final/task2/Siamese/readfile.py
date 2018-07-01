import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

"""

total images for training: 500 * 80

"""
if __name__ == "__main__":

    """
    path = "../task2-dataset/base/"
    folders = list()

    X_train = np.zeros((80,500,32,32,3)).astype(np.float32)
    Y_train = np.zeros((80,500,1))
    X_test = np.zeros((80,100,32,32,3)).astype(np.float32)
    Y_test = np.zeros((80,100,1))

    for folder in listdir(path):
        folders.append(folder)

    folders = sorted(folders)


    label = 0

    for folder in folders:
        train_path = path + folder + "/train/"
        test_path = path + folder + "/test/"
        i = 0
        j = 0
        for img in listdir(train_path):
            img_path = train_path + img
            X_train[label][i] = cv2.imread(img_path)
            Y_train[label][i] = label
            i += 1
        for img in listdir(test_path):
            img_path = test_path + img
            X_test[label][j] = cv2.imread(img_path)
            Y_test[label][j] = label
            j += 1
        label += 1

    X_train = (X_train - 127.5) / 127.5
    X_test = (X_test - 127.5) / 127.5

    X_train = X_train.transpose(0,1,4,2,3)
    X_test = X_test.transpose(0,1,4,2,3)

    np.save("../data/tmp/X_base_train.npy",X_train)
    np.save("../data/tmp/Y_base_train.npy",Y_train)
    np.save("../data/tmp/X_base_test.npy",X_test)
    np.save("../data/tmp/Y_base_test.npy",Y_test)

    print ("Finish loading!")
    """
    # ===== novel classes =====
    
    path = "../task2-dataset/novel/"
    folders = list()
    classes = list()

    X_novel = np.zeros((20,500,32,32,3)).astype(np.float32)
    Y_novel = np.zeros((20,500,1))
    #X_test = np.zeros((2000,32,32,3)).astype(np.float32)

    for folder in listdir(path):
        folders.append(folder)

    folders = sorted(folders)

    label = 0

    for folder in folders:
        train_path = path + folder + "/train/"
        classes.append(folder[6:8])
        i = 0
        img_tmp = list()

        for img in listdir(train_path):
            img_tmp.append(img)

        img_tmp = sorted(img_tmp)

        for img in img_tmp:
            print (folder,img)
            img_path = train_path + img
            X_novel[label][i] = cv2.imread(img_path)
            Y_novel[label][i] = label
            i += 1
        label += 1

    X_novel = (X_novel - 127.5) / 127.5
    X_novel = X_novel.transpose(0,1,4,2,3)

    np.save("../data/tmp/X_novel_sort.npy",X_novel)
    np.save("../data/tmp/Y_novel_sort.npy",Y_novel)
    np.save("../data/class_table.npy",np.array(classes))
    """

    path = "../test/"
    test = list()
    X_test = np.zeros((2000,32,32,3))

    for index in range(2000):
        img_path = path + str(index) + ".png"
        print (img_path)
        X_test[index] = cv2.imread(img_path)

    X_test = (X_test -127.5) / 127.5
    X_test = X_test.transpose(0,3,1,2)
    np.save("../data/tmp/X_test.npy",X_test)

    """


