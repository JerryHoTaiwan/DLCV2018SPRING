from torch.utils.data import Dataset
from scipy.misc import imread
import numpy as np
import os


class Cifar100(Dataset):
    def __init__(self, args, file='novel', mode="test"):
        super(Cifar100, self).__init__()
        self.args = args
        self.file = file
        self.mode = mode
        self.image = []
        self.label = []
        self.load_file()

    def load_file(self):
        if self.file == 'novel' and self.mode == 'test':
            self.image = [imread(os.path.join(self.args.test_dir, image))
                          for image in sorted(os.listdir(self.args.test_dir))]

        else:
            base = os.path.join(self.args.train_dir, self.file)
            classes_dir = [os.path.join(base, files) for files in sorted(os.listdir(base)) if files.startswith('class')]
            for idx, dir in enumerate(classes_dir):
                train_path = os.path.join(dir, self.mode)
                image = [np.expand_dims(imread(os.path.join(train_path, image)), axis=2)
                         for image in sorted(os.listdir(train_path)) if image.endswith(".png")]
                label = [idx for _ in range(len(self.image))]

                self.image.extend(image)
                self.label.extend(label)

    def __getitem__(self, index):
        if self.file == 'novel' and self.mode == 'test':
            return self.image[index]
        else:
            return self.image[index], self.label[index]

    def __len__(self):
        return len(self.image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/task2-dataset')
    parser.add_argument('--test-dir', default='../datasets/test')
    data = Cifar100(parser.parse_args())
