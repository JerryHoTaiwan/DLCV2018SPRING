from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch

shuffle_index = np.random.permutation(80)

class MyDataset(Dataset):
    def __init__(self):
        #self.X_train = np.load("../augment_data/aug_all.npy")#.reshape(-1,500,3,32,32)
        self.X_train = np.load("../data/tmp/X_base_train.npy")#.reshape(-1,100,3,32,32)
        self.num_sample = 79

    def __getitem__(self, idx):
        if (idx % 2 == 0):
            index = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_train[shuffle_index[index]][random.randint(0, 499)]),
                    torch.FloatTensor(self.X_train[shuffle_index[index]][random.randint(0, 499)]),
                    torch.FloatTensor([1]))

        else:
            index1 = random.randint(0, self.num_sample)
            index2 = random.randint(0, self.num_sample)
            while index1 == index2:
                index2 = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_train[shuffle_index[index1]][random.randint(0, 499)]),
                    torch.FloatTensor(self.X_train[shuffle_index[index2]][random.randint(0, 499)]),
                    torch.FloatTensor([0]))

    def __len__(self):
        return 100000


class ValDataset(Dataset):
    def __init__(self):
        self.X_valid = np.load("../data/tmp/X_base_test.npy")#.reshape(-1,100,3,32,32)
        self.num_sample = 79

    def __getitem__(self, idx):
        if idx % 2:
            index = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_valid[shuffle_index[index]][random.randint(0, 99)]),
                    torch.FloatTensor(self.X_valid[shuffle_index[index]][random.randint(0, 99)]),
                    torch.FloatTensor([1]))

        else:
            index1 = random.randint(0, self.num_sample)
            index2 = random.randint(0, self.num_sample)
            while index1 == index2:
                index2 = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_valid[shuffle_index[index1]][random.randint(0, 99)]),
                    torch.FloatTensor(self.X_valid[shuffle_index[index2]][random.randint(0, 99)]),
                    torch.FloatTensor([0]))

    def __len__(self):
        return 10000

class NovelDataset(Dataset):
    def __init__(self):
        self.X_novel = np.load("../data/tmp/X_novel_sort.npy")#.reshape(-1,100,3,32,32)
        self.num_sample = 24

    def __getitem__(self, idx):
        if idx % 2:
            index = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_novel[index][random.randint(0, 99)]),
                    torch.FloatTensor(self.X_novel[index][random.randint(0, 99)]),
                    torch.FloatTensor([1]))

        else:
            index1 = random.randint(0, self.num_sample)
            index2 = random.randint(0, self.num_sample)
            while index1 == index2:
                index2 = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_novel[index1][random.randint(0, 99)]),
                    torch.FloatTensor(self.X_novel[index2][random.randint(0, 99)]),
                    torch.FloatTensor([0]))

    def __len__(self):
        return 10000

class FewDataset(Dataset):
    def __init__(self,k=5,seed=1125):
        self.X_novel = np.load("../data/tmp/X_novel_sort.npy")#.reshape(-1,100,3,32,32)
        self.k = k
        self.seed = seed
        self.X_few = np.zeros((20,k,3,32,32)).astype(np.float32)

        random.seed(self.seed)
        #sample = [random.randint(0, 79) for _ in range(k)]
        sample = [random.randint(0, 499) for _ in range(k)]
        
        print (self.X_novel.shape)

        for i in range(len(self.X_novel)):
            for j in range(k):
                self.X_few[i][j] = self.X_novel[i][sample[j]]
        self.X_few = torch.from_numpy(self.X_few)

        self.num_sample = 19

    def __getitem__(self, idx):
        if idx % 2:
            index = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_few[index][random.randint(0, self.k-1)]),
                    torch.FloatTensor(self.X_few[index][random.randint(0, self.k-1)]),
                    torch.FloatTensor([1]))

        else:
            index1 = random.randint(0, self.num_sample)
            index2 = random.randint(0, self.num_sample)
            while index1 == index2:
                index2 = random.randint(0, self.num_sample)
            return (torch.FloatTensor(self.X_few[index1][random.randint(0, self.k-1)]),
                    torch.FloatTensor(self.X_few[index2][random.randint(0, self.k-1)]),
                    torch.FloatTensor([0]))

    def __len__(self):
        return 100000