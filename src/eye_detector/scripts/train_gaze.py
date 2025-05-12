from argparse import ArgumentParser

import torch
import torch.jit

from eye_detector.train_gaze.dataset import create_dataset
from eye_detector.train_gaze.model import create_new_net
from eye_detector.train_gaze.train import train
from eye_detector.train_gaze.test import test_data


parser = ArgumentParser(description="Train gaze")
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--size-ratio", default=1.0, type=float)
parser.add_argument("--output", default="outdata/gaze.pth")


def __save(net, output: str):
    print("Saving to", output, "…")
    torch.save(net.state_dict(), output)


def main(args):
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda")
    print("Preparing dataset…")
    trainset, testset = create_dataset(device, args.size_ratio)

    print("JITing neural network…")
    net = create_new_net(device)

    print("Training…")
    train(trainset, testset, test_data, net, max_epoch=args.epoch, save_cb=lambda net: __save(net, args.output))
    print('Finished Training')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
