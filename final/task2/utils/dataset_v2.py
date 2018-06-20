from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from scipy.misc import imread
import numpy as np
import torch
import random
import os


class Cifar100Task(object):
    def __init__(self, args):
        super(Cifar100Task, self).__init__()
        folders = [os.path.join(args.train_dir, folder)
                   for folder in sorted(os.listdir(args.train_dir)) if folder.startswith('class')]
        self.num_classes = 20
        self.train_num = 5
        self.test_num = 10

        class_folders = random.sample(folders, self.num_classes)

        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))

        self.train = []
        self.test = []
        samples = dict()

        for c in class_folders:
            c = os.path.join(c, 'train')
            temp = [os.path.join(c, x) for x in os.listdir(c)]

            samples[c] = random.sample(temp, len(temp))

            self.train += samples[c][:self.train_num]
            self.test += samples[c][self.train_num:self.train_num + self.test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train]
        self.test_labels = [labels[self.get_class(x)] for x in self.test]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])


class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train if self.split == 'train' else self.task.test
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, index):
        raise NotImplementedError("Abstract class only, please implement customize dataset.")

class Cifar100(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(Cifar100, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image_root = self.image_roots[index]
        image = imread(image_root)
        if self.transform is not None:
            image = self.transform(image)  # 3 x 32 x 32
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        # print(num_per_class, num_cl, num_inst)

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_mini_imagenet_data_loader(task, num_per_class=5, split='train',shuffle = False):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = Cifar100(task,split=split,transform=transforms.Compose([transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

if __name__ == "__main__":
    import argparse
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/task2-dataset/base')
    parser.add_argument('--test-dir', default='../datasets/test')
    data = Cifar100Task(parser.parse_args())

    data_loader = get_mini_imagenet_data_loader(data)

    samples, sample_labels = data_loader.__iter__().next()  # 100*3*84*84
    print(samples)
    samples, sample_labels = data_loader.__iter__().next()  # 100*3*84*84
    print(samples)
