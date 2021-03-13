from time import time
from argparse import ArgumentParser

from eye_detector.gen_to_train.dump import recreate_directory
from eye_detector.gen_to_train.dump import Dumper
from eye_detector.gen_to_train.transforms import HogEye

parser = ArgumentParser(
    description="Generate data to learning detection of eye",
)
parser.add_argument('--chunk-size', type=int, default=1000)
parser.add_argument('-j', '--jobs', type=int, default=8)
parser.add_argument('--room-multipler', type=int, default=1)
parser.add_argument('--face-multipler', type=int, default=1)
parser.add_argument(
    '--transform',
    default='hog',
    choices=['hog', 'lbp', 'no-transform'],
)


if __name__ == "__main__":
    args = parser.parse_args()

    transform_eye = HogEye()
    dumper = Dumper(transform_eye, args)
    recreate_directory()

    t = time()
    dumper.dump()
    diff = time() - t
    print("TIME:", diff)
    print("GENERATION DONE")
