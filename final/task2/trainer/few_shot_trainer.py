import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.dataset_v2 import get_mini_imagenet_data_loader, Cifar100Task
from model import Embedder, Relation
import torch.nn as nn
import torch
import numpy as np
import csv
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
        self.embedder = Embedder().cuda() if self.with_cuda else Embedder()
        self.relation = Relation().cuda() if self.with_cuda else Relation()
        self.criterion = nn.MSELoss().cuda() if self.with_cuda else nn.MSELoss()
        self.embedder_optimizer = Adam(self.embedder.parameters(), lr=0.001)
        self.embedder_scheduler = StepLR(self.embedder_optimizer, step_size=100000, gamma=0.5)
        self.relation_optimizer = Adam(self.relation.parameters(), lr=0.001)
        self.relation_scheduler = StepLR(self.relation_optimizer, step_size=100000, gamma=0.5)


    def train(self):
        self.embedder.train()
        self.relation.train()

        for episode in range(1, self.args.episodes + 1):
            total_loss, total_acc = 0, 0
            self.embedder_scheduler.step(epoch=episode)
            self.relation_scheduler.step(epoch=episode)

            task = Cifar100Task(self.args)
            sample_dataloader = get_mini_imagenet_data_loader(task, 5, 'train', shuffle=False)
            batch_dataloader = get_mini_imagenet_data_loader(task, 10, 'test', shuffle=True)

            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()



            sample_features = self.embedder(Variable(samples).cuda() if self.with_cuda else Variable(samples))

            sample_features = sample_features.view(20, 5, 64, 15, 15)
            sample_features = torch.sum(sample_features, dim=1).squeeze(1)
            batch_features = self.embedder(Variable(batches).cuda() if self.with_cuda else Variable(batches))


            sample_features_ext = sample_features.unsqueeze(0).repeat(10 * 20, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(20, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1, 128, 15, 15)

            relations = self.relation(relation_pairs).view(-1, 20) # 200 * 20
           # print(relations)

            #print(sample_features_ext) # 200*20*64*6*6
            one_hot_labels = Variable(torch.zeros(10 * 20, 20).scatter_(1, batch_labels.view(-1, 1), 1)).cuda()

            loss = self.criterion(relations, one_hot_labels)
            result = torch.max(relations, dim=1)[1]
            accuracy = np.mean((result.cpu() == batch_labels).data.numpy())
            #print(batch_features_ext) # 200*20*64*6*6

            loss.backward()
            self.embedder_optimizer.step()
            self.relation_optimizer.step()


            torch.nn.utils.clip_grad_norm(self.embedder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(self.relation.parameters(), 0.5)


            if episode%100 == 0:
                print(relations)

            print('Episode: {}/{} Loss: {:.6f} Acc: {:.0f}/200'.format(episode,
                                                       self.args.episodes,
                                                       loss.data[0],
                                                       200*accuracy), end='\r')
            sys.stdout.write('\033[K')

    def __eval(self):
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (in_fig, _) in enumerate(self.test_data_loader):
                in_fig = Variable(in_fig).cuda() if self.with_cuda else Variable(in_fig)
                output = self.model(in_fig)
                result = torch.max(output, dim=1)[1]

            with open('result_csv/test.csv', 'w') as f:
                s = csv.writer(f, delimiter=',', lineterminator='\n')
                s.writerow(["image_id", "predicted_label"])
                for idx, predict_label in enumerate(result.cpu().data.numpy().tolist()):
                    s.writerow([idx, predict_label])
            print("Saving inference label csv as result/test.csv")

    def __save_checkpoint(self, epoch, current_loss):
        state = {
            'model': 'FashionMNIST CNN',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list
        }

        if not os.path.exists("checkpoints/fashion_mnist"):
            os.makedirs("checkpoints/fashion_mnist")

        filename = "checkpoints/fashion_mnist/epoch{}_checkpoint.pth.tar".format(epoch)
        best_filename = "checkpoints/fashion_mnist/best_checkpoint.pth.tar"

        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
        if self.min_loss > current_loss:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, self.min_loss, current_loss))
            self.min_loss = current_loss


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
