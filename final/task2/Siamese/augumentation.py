from dataloader import my_transform, Cifar100
from skimage import io
import numpy as np
import sys

base_class_path = sys.argv[1] # '../data/task2-dataset/base/'
output_dir = sys.argv[2] # 'output/'
# base_class_path = '../data/task2-dataset/base/'
# output_dir = 'npy/'
train_dataset = Cifar100(base_class_path, 'train', transform=my_transform)
img_dict = {}
for i, (img, label) in enumerate(train_dataset):
    # if i > 1000-1:
    #     break
    img_np = np.array(img)
    if label not in img_dict:
        img_dict[label] = [img_np]
    else:
        img_dict[label].append(img_np)

for j in range(3):
    for i, (img, label) in enumerate(train_dataset):
        img_dict[label].append(np.array(img))

# print(img_dict)
img_big_list = []
for label in img_dict:
    # print(label, np.array(img_dict[label]).shape)
    # np.save(output_dir+'/aug_%02d.npy'%label, img_dict[label])
    img_big_list.append(img_dict[label])
# (80, 2000, 32, 32, 3)
np.save(output_dir+'/aug_all.npy', np.rollaxis(np.array(img_big_list), 4, 2))
print(np.load(output_dir+'/aug_all.npy').shape)