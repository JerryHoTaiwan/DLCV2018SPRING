import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from utils.dataset import FashionMNIST
from model.models import CNN
import torch.nn as nn
import torch
import numpy as np
import csv
import sys
import os


class TestTrainer:
    def __init__(self, args):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        self.__load_file()
        self.__build_model()
        # self.__init_weight()

        self.min_loss = float('inf')
        self.loss_list, self.acc_list = [], []
        self.val_loss_list, self.val_acc_list = [], []

    def __load_file(self):
        self.train_dataset = datasets.FashionMNIST("datasets/full_fashion", train=True, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor()
                                                   ]))
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True)
        self.test_dataset = FashionMNIST(self.args,
                                         mode='train',
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        self.test_data_loader = DataLoader(dataset=self.test_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        print("Full FashionMnist Loaded")

    def __build_model(self):
        self.model = CNN().cuda() if self.with_cuda else CNN()
        self.criterion = nn.CrossEntropyLoss().cuda() if self.with_cuda else nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def __init_weight(self):
        checkpoint = torch.load(self.args.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])

        # initialize classifier weight from zero
        for m in self.model.classifier:
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

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
                                                                 total_acc / len(self.train_data_loader)), end=' ')

            self.loss_list.append(total_loss / len(self.train_data_loader))
            self.acc_list.append(total_acc / len(self.train_data_loader))

            if self.args.verbosity:
                self.__eval()
            else:
                print()

            self.__save_checkpoint(epoch, total_loss / len(self.train_data_loader))

    def __eval(self):
        with torch.no_grad():
            self.model.eval()
            total_loss, total_acc = 0, 0
            for batch_idx, (in_fig, label) in enumerate(self.test_data_loader):
                in_fig = Variable(in_fig).cuda() if self.with_cuda else Variable(in_fig)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)

                output = self.model(in_fig)
                loss = self.criterion(output, label)

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.data[0]
                total_acc += accuracy

            print('valid_loss: {:.6f}  valid_acc: {:.6f}'.format(total_loss / len(self.test_data_loader),
                                                                 total_acc / len(self.test_data_loader)))

        return total_loss / len(self.test_data_loader), total_acc / len(self.test_data_loader)
            # with open('result_csv/test_full_dataset.csv', 'w') as f:
            #     s = csv.writer(f, delimiter=',', lineterminator='\n')
            #     s.writerow(["image_id", "predicted_label"])
            #     for idx, predict_label in enumerate(result.cpu().data.numpy().tolist()):
            #        s.writerow([idx, predict_label])
            #print("Saving inference label csv as result/test_full_dataset.csv")




    def __save_checkpoint(self, epoch, current_loss):
        state = {
            'model': 'FashionMNIST CNN',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list
        }

        if not os.path.exists("checkpoints/full_fashion_mnist"):
            os.makedirs("checkpoints/full_fashion_mnist")

        filename = "checkpoints/full_fashion_mnist/epoch{}_checkpoint.pth.tar".format(epoch)
        best_filename = "checkpoints/full_fashion_mnist/best_checkpoint.pth.tar"

        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
        if self.min_loss > current_loss:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, self.min_loss, current_loss))
            self.min_loss = current_loss
