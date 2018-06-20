import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.dataset_v2 import get_mini_imagenet_data_loader, Cifar100Task
from model.relation_network import RelationNetwork
import torch.nn as nn
import torch
import scipy.stats
import numpy as np
import sys
import os


class FewshotTrainer:
    def __init__(self, args):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        #self.__load()
        self.__build_model()

        self.min_loss = float('inf')
        self.loss_list, self.acc_list = [], []
        self.val_loss_list, self.val_acc_list = [], []

    def __build_model(self):
        print("Building Model.......")
        self.model = RelationNetwork().cuda() if self.with_cuda else RelationNetwork()
        self.criterion = nn.MSELoss().cuda() if self.with_cuda else nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=100000, gamma=0.5)

    def train(self):
        for episode in range(1, self.args.episodes + 1):
            self.model.train()
            self.scheduler.step(epoch=episode)

            task = Cifar100Task(self.args, mode='train')
            sample_dataloader = get_mini_imagenet_data_loader(task, 5, 'train', shuffle=False)
            batch_dataloader = get_mini_imagenet_data_loader(task, 10, 'test', shuffle=True)

            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()

            samples = Variable(samples).cuda() if self.with_cuda else Variable(samples)
            batches = Variable(batches).cuda() if self.with_cuda else Variable(batches)

            relations = self.model(samples, batches)

            one_hot_labels = Variable(torch.zeros(10 * 20, 20).scatter_(1, batch_labels.view(-1, 1), 1)).cuda()

            result = torch.max(relations, dim=1)[1]
            accuracy = np.mean((result.cpu() == batch_labels).data.numpy())

            loss = self.criterion(relations, one_hot_labels)
            loss.backward()
            self.optimizer.step()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            print('Episode: {}/{} Loss: {:.6f} Acc: {:.3f}'.format(episode,
                                                                   self.args.episodes,
                                                                   loss.data[0],
                                                                   accuracy), end='\r')
            if episode % self.args.eval_freq == 0:
                print()
                self.__eval()
            else:
                sys.stdout.write('\033[K')

            if episode % self.args.save_freq == 0:
                self.__save_checkpoint(episode, loss.data[0])

            self.loss_list.append(loss.data[0])
            self.acc_list.append(accuracy)


    def __eval(self):
        with torch.no_grad():
            self.model.eval()
            total_acc = []
            for episode in range(self.args.test_episodes):
                accuracy = 0
                task = Cifar100Task(self.args, mode='test')
                sample_dataloader = get_mini_imagenet_data_loader(task, 5, 'train', False)
                test_dataloader = get_mini_imagenet_data_loader(task, 10, 'test', True)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                for test_images, test_labels in test_dataloader:
                    sample_images = Variable(sample_images).cuda() if self.with_cuda else Variable(sample_images)
                    test_images = Variable(test_images).cuda() if self.with_cuda else Variable(test_images)
                    relations = self.model(sample_images, test_images)

                    result = torch.max(relations, dim=1)[1]
                    acc = np.mean((result.cpu() == test_labels).data.numpy())
                    accuracy += acc

                accuracy = accuracy / 200
                total_acc.append(accuracy)

                print('Validation Episode: {}/{} Acc: {:.3f}'.format(episode,
                                                          600,
                                                          accuracy), end='\r')
                sys.stdout.write('\033[K')

            test_accuracy, h = mean_confidence_interval(total_acc)
            print("Test accuracy: {:.6f}, h: {:.6f}".format(test_accuracy, h))


    def __save_checkpoint(self, episode, current_loss):
        state = {
            'model': '{}-shot learning'.format(5),
            'epoch': episode,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list
        }

        if not os.path.exists("checkpoints/5-shot"):
            os.makedirs("checkpoints/5-shot")

        filename = "checkpoints/5-shot/episode{}_checkpoint.pth.tar".format(episode)
        best_filename = "checkpoints/5-shot/best_checkpoint.pth.tar"

        if episode % self.args.save_freq == 0:
            torch.save(state, f=filename)
        if self.min_loss > current_loss:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(episode, self.min_loss, current_loss))
            self.min_loss = current_loss


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/task2-dataset/base')
    parser.add_argument('--test-dir', default='../datasets/test')
    parser.add_argument('--episodes', default=10)
    parser.add_argument('--cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    test = FewshotTrainer(parser.parse_args())
    test.train()
