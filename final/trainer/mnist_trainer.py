import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from model.models import CNN
import torch.nn as nn
import torch
import numpy as np
import sys
import os


class MnistTrainer:
    def __init__(self, args):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        self.__load_file()
        self.__build_model()

        self.min_loss = float('inf')
        self.loss_list = []
        self.acc_list = []

    def __load_file(self):
        """
        self.train_dataset = FashionMNIST(self.args,
                                          mode='train',
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
        self.test_dataset = FashionMNIST(self.args,
                                         mode='test',
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        self.test_data_loader = DataLoader(dataset=self.test_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=False)
        """
        # MNIST for transfer learning
        self.train_dataset = datasets.MNIST("datasets/mnist", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)

        self.test_dataset = datasets.MNIST("datasets/mnist", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))
        self.test_data_loader = DataLoader(dataset=self.test_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=False)

    def __build_model(self):
        self.model = CNN().cuda() if self.with_cuda else CNN()
        self.criterion = nn.CrossEntropyLoss().cuda() if self.with_cuda else nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            total_loss, total_acc = 0, 0
            for batch_idx, (in_fig, label) in enumerate(self.train_data_loader):
                in_fig = Variable(in_fig).cuda() if self.with_cuda else Variable(in_fig)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)

                self.optimizer.zero_grad()
                output = self.model(in_fig)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.data[0]
                total_acc += accuracy

                if batch_idx % self.args.log_step == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: {:.6f} Acc: {:.6f}'.format(
                        epoch,
                        self.args.epochs,
                        batch_idx * self.train_data_loader.batch_size,
                        len(self.train_data_loader) * self.train_data_loader.batch_size,
                        100.0 * batch_idx / len(self.train_data_loader),
                        loss.data[0],
                        accuracy
                    ), end='\r')
                    sys.stdout.write('\033[K')
            print("Epoch: {}/{} Loss: {:.6f} Acc: {:.6f}".format(epoch,
                                                                 self.args.epochs,
                                                                 total_loss / len(self.train_data_loader),
                                                                 total_acc / len(self.train_data_loader)))

            self.loss_list.append(total_loss / len(self.train_data_loader))
            self.acc_list.append(total_acc / len(self.train_data_loader))
            self.__save_checkpoint(epoch, total_loss / len(self.train_data_loader))

    def __save_checkpoint(self, epoch, current_loss):
        state = {
            'model': 'MNIST CNN',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list
        }

        if not os.path.exists("checkpoints/mnist"):
            os.makedirs("checkpoints/mnist")

        filename = "checkpoints/mnist/epoch{}_checkpoint.pth.tar".format(epoch)
        best_filename = "checkpoints/mnist/best_checkpoint.pth.tar"

        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
        if self.min_loss > current_loss:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, self.min_loss, current_loss))
            self.min_loss = current_loss
