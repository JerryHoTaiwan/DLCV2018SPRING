from dataloader import my_transform, train_dataset, Cifar100
from skimage import io
import numpy as np
import sys

base_class_path = sys.argv[1] # '../data/task2-dataset/base/'
output_dir = sys.argv[2] # 'output/'
train_dataset = Cifar100(base_class_path, 'train', transform=my_transform)
img_dict = {}
for i, (img, label) in enumerate(train_dataset):
    img_np = np.array(img)
    if label not in img_dict:
        img_dict[label] = [img_np]
    else:
        img_dict[label].append(img_np)

for j in range(3):
    for i, (img, label) in enumerate(train_dataset):
        img_dict[label].append(np.array(img))

# print(img_dict)
for label in img_dict:
    # print(label, np.array(img_dict[label]).shape)
    np.save(output_dir+'/aug_%02d.npy'%label, img_dict[label])