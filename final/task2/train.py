from trainer.few_shot_trainer import FewshotTrainer
import argparse


def main(args):
    trainer = FewshotTrainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Final Project")
    parser.add_argument('--train-dir', default='datasets/task2-dataset/base',
                        help='training data directory')
    parser.add_argument('--test-dir', default='datasets/test',
                        help='testing data directory')
    parser.add_argument('--episodes', default=500000, type=int,
                        help='training epochs (default: 500000)')
    parser.add_argument('--test-episodes', default=600, type=int,
                        help='testing episodes (default: 600)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--verbosity', action='store_true',
                        help='evaluation visualization (output csv)')
    parser.add_argument('--save-freq', default=2000, type=int,
                        help='save checkpoints frequency (default: 1)')
    parser.add_argument('--eval-freq', default=10000, type=int,
                        help='evaluation frequency for validation')
    main(parser.parse_args())
