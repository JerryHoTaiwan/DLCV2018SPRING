from utils.dataset import FashionMNIST
from model.models import CNN
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
import argparse
import csv


def main(args):
    checkpoint = torch.load(args.checkpoint)
    with_cuda = not args.no_cuda

    model = CNN().cuda() if with_cuda else CNN()
    model.load_state_dict(checkpoint['state_dict'])

    test_dataset = FashionMNIST(args,
                                mode='test',
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=len(test_dataset),
                                  shuffle=False)
    with torch.no_grad():
        model.eval()
        for batch_idx, (in_fig, _) in enumerate(test_data_loader):
            in_fig = Variable(in_fig).cuda() if with_cuda else Variable(in_fig)
            output = model(in_fig)
            result = torch.max(output, dim=1)[1]

        with open('result.csv', 'w') as f:
            s = csv.writer(f, delimiter=',', lineterminator='\n')
            s.writerow(["image_id", "predicted_label"])
            for idx, predict_label in enumerate(result.cpu().data.numpy().tolist()):
                s.writerow([idx, predict_label])
        print("Saving inference label csv as result.csv")

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='datasets/Fashion_MNIST_student/train')
    parser.add_argument('--test-dir', default='datasets/Fashion_MNIST_student/test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fashion_mnist/best_checkpoint.pth.tar')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())
