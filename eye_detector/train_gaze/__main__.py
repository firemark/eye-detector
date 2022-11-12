from argparse import ArgumentParser

import torch
import torch.jit

from eye_detector.train_gaze.dataset import create_dataset, WIDTH, HEIGHT
from eye_detector.train_gaze.model import Net
from eye_detector.train_gaze.train import train
from eye_detector.train_gaze.test import test_data


parser = ArgumentParser(description="Train gaze")
parser.add_argument("--epoch", default=10, type=int)


def main(args):
    trainset, testset = create_dataset()

    net = Net()
    example_input = [(torch.rand(1, 9), torch.rand(1, 3, WIDTH, HEIGHT))]
    net = torch.jit.trace(net, example_input)
    train(trainset, testset, test_data, net, max_epoch=args.epoch)
    torch.save(net.state_dict(), "outdata/net.pth")
    print('Finished Training')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
