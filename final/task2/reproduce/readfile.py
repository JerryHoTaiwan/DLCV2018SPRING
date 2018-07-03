import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

    # ===== novel classes =====
    
def load_novel(path):

    print ("loading novel classes...")

    #path = "../task2-dataset/novel/"
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
            img_path = train_path + img
            X_novel[label][i] = cv2.imread(img_path)
            Y_novel[label][i] = label
            i += 1
        label += 1

    X_novel = (X_novel - 127.5) / 127.5
    X_novel = X_novel.transpose(0,1,4,2,3)

    return (X_novel,np.array(classes))

def load_test(path):
    print ("loading testing set...")

    #path = "../test/"
    test = list()
    X_test = np.zeros((2000,32,32,3))

    for index in range(2000):
        img_path = path + str(index) + ".png"
        X_test[index] = cv2.imread(img_path)

    X_test = (X_test -127.5) / 127.5
    X_test = X_test.transpose(0,3,1,2)

    return X_test



