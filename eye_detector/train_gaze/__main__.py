from argparse import ArgumentParser

import torch

from eye_detector.train_gaze.dataset import create_dataset
from eye_detector.train_gaze.model import Net
from eye_detector.train_gaze.train import train
from eye_detector.train_gaze.test import test_data


parser = ArgumentParser(description="Train gaze")
parser.add_argument("--epoch", default=10, type=int)


def main(args):
    trainset, testset = create_dataset()

    net = Net()
    train(trainset, net, max_epoch=args.epoch)
    test_data(testset, net)
    torch.save(net.state_dict(), "outdata/net.pth")
    print('Finished Training')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
