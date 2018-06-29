import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import numpy as np

my_transform = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4),
    transforms.RandomResizedCrop(32),
    transforms.RandomRotation(20)
    # transforms.ToTensor()
])
# my_imgfolder = ImageFolder('../data/task2-dataset/base/', transform=my_transform)

class Cifar100(Dataset):
    def __init__(self, root_dir, train_or_test, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = []
        label = 0
        for class_dir in os.listdir(self.root_dir):
            for img_name in os.listdir(root_dir+'/'+class_dir+'/'+train_or_test+'/'):
                self.img_list.append((root_dir+'/'+class_dir+'/'+train_or_test+'/'+img_name, label))
            label += 1
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img_name, label = self.img_list[idx]
        image = io.imread(img_name)
        if self.transform:
            sample = self.transform(image)
        else:
            sample = image
        return sample, label

class Cifar100_npy(Dataset):
    def __init__(self, x_npy, y_npy, transform=None):
        self.x_npy = np.load(x_npy)
        self.y_npy = np.load(y_npy)
        self.transform = transform
    def __len__(self):
        return len(self.x_npy)
    def __getitem__(self, idx):
        image = self.x_npy[idx]
        if self.transform:
            sample = self.transform(image)
        else:
            sanple = image
        return sample, y_npy[idx]

train_dataset = Cifar100('../data/task2-dataset/base/', 'train', transform=my_transform)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

train_dataset_npy = Cifar100_npy('npy/X_base_train.npy', 'npy/Y_base_train.npy', transform=my_transform)
train_dataloader_npy = DataLoader(train_dataset_npy, batch_size=256, shuffle=True, num_workers=4)
if __name__ == '__main__':
    # for (x, y) in my_dataset:
    #     print(x, y)
    for i_batch, batch in enumerate(train_dataloader):
        print(i_batch)
        print(batch[1])
        if i_batch>1:
            exit()