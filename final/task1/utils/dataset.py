from torch.utils.data import Dataset
from scipy.misc import imread
import numpy as np
import os


class FashionMNIST(Dataset):
    def __init__(self, args, mode='train', transform=None):
        super(FashionMNIST, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = transform

        self.image = []
        self.label = []
        self.__load_data()

    def __load_data(self):
        if self.mode == 'train':
            base = self.args.train_dir
            classes_dir = [os.path.join(base, file) for file in sorted(os.listdir(base)) if not file.startswith('.DS')]
            for idx, dir in enumerate(classes_dir):
                images = [np.expand_dims(imread(os.path.join(dir, image)), axis=2)
                          for image in sorted(os.listdir(dir)) if image.endswith(".png")]
                labels = [idx for _ in range(len(images))]
                self.image.extend(images)
                self.label.extend(labels)
        else:
            base = self.args.test_dir
            self.image = [np.expand_dims(imread(os.path.join(base, image)), axis=2)
                          for image in sorted(os.listdir(base), key=lambda x: int(x[:-4])) if image.endswith(".png")]
            self.label = [0 for _ in range(len(self.image))]

    def __getitem__(self, index):
        image = self.transform(self.image[index]) if self.transform is not None else self.image[index]
        label = self.label[index]
        return image, label

    def __len__(self):
        return len(self.image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/Fashion_MNIST_student/train')
    parser.add_argument('--test-dir', default='../datasets/Fashion_MNIST_student/test')
    data = FashionMNIST(parser.parse_args())
