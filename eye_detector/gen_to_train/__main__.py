from time import time
from argparse import ArgumentParser

from eye_detector.gen_to_train.dump import recreate_directory
from eye_detector.gen_to_train.dump import Dumper
from eye_detector.gen_to_train.transforms import TRANSFORMS
from eye_detector.model import store_transform

parser = ArgumentParser(
    description="Generate data to learning detection of eye",
)
parser.add_argument('--chunk-size', type=int, default=1000)
parser.add_argument('-j', '--jobs', type=int, default=8)
parser.add_argument('--room-multipler', type=float, default=1.0)
parser.add_argument('--face-multipler', type=float, default=1.0)
parser.add_argument('--face-as-unique-label', default=False, action="store_true")
parser.add_argument(
    '-t', '--transform',
    default='hog-16-8',
    choices=list(TRANSFORMS.keys()),
)


if __name__ == "__main__":
    args = parser.parse_args()

    recreate_directory()
    transform = TRANSFORMS[args.transform]
    dumper = Dumper(transform, args)

    t = time()
    dumper.dump()
    diff = time() - t
    print("TIME:", diff)

    print("GENERATION DONE")

    store_transform(transform)
    print("TRANSFORMATION SAVED")
