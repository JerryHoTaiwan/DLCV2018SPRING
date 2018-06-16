from trainer.mnist_trainer import MnistTrainer
from trainer.fashion_mnist_trainer import FashionmnistTrainer
import argparse


def main(args):
    trainer = eval(args.dataset.title() + 'Trainer')(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Final Project")
    parser.add_argument('--dataset', default='FashionMnist', type=str,
                        help='training architecture [Mnist]')
    parser.add_argument('--train-dir', default='datasets/Fashion_MNIST_student/train',
                        help='training data directory')
    parser.add_argument('--test-dir', default='datasets/Fashion_MNIST_student/test',
                        help='testing data directory')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size of the model (default: 128)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='training epochs (default: 100)')
    parser.add_argument('--log-step', default=1, type=int,
                        help='printing step size (default: 1')
    parser.add_argument('--checkpoint', default='checkpoints/mnist/epoch50_checkpoint.pth.tar',
                        help='load initialize weight from checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--verbosity', action='store_true',
                        help='evaluation visualization (output csv)')
    parser.add_argument('--save-freq', default=1, type=int,
                        help='save checkpoints frequency (default: 1)')
    main(parser.parse_args())
