from time import time
from argparse import ArgumentParser

from eye_detector.gen_to_train.dump import recreate_directory
from eye_detector.gen_to_train.dump import Dumper
from eye_detector.gen_to_train.transforms import TRANSFORMS
from eye_detector.gen_to_train.image_transforms import IMAGE_TRANSFORMS
from eye_detector.gen_to_train import data_loader
from eye_detector.model import store_transform


DATASETS = {
    'mrl': data_loader.MrlEyeDataLoader,
    'synth': data_loader.SynthEyeDataLoader,
    'helen': data_loader.HelenEyeDataLoader,
}

parser = ArgumentParser(
    description="Generate data to learning detection of eye",
)
parser.add_argument('--chunk-size', type=int, default=1000)
parser.add_argument('-j', '--jobs', type=int, default=8)
parser.add_argument('--room-multipler', type=float, default=1.0)
parser.add_argument('--face-multipler', type=float, default=1.0)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument(
    '-d', '--dataset',
    default='mrl',
    choices=list(DATASETS.keys()),
)
parser.add_argument('--face-as-unique-label', default=False, action="store_true")
parser.add_argument(
    '-t', '--transform',
    default='hog-16-8',
    choices=list(TRANSFORMS.keys()),
)
parser.add_argument(
    '-i', '--image-transform',
    default='gray',
    choices=list(IMAGE_TRANSFORMS.keys()),
)


if __name__ == "__main__":
    args = parser.parse_args()
    eye_data_cls = DATASETS[args.dataset]
    eye_data_cls.assert_args(args)

    args.noise *= 0.01

    recreate_directory()
    image_transform = IMAGE_TRANSFORMS[args.image_transform]
    transform = TRANSFORMS[args.transform]
    transform.set_image_transform(image_transform)
    dumper = Dumper(transform, args, eye_data_cls)

    t = time()
    dumper.dump()
    diff = time() - t
    print("GENERATION DONE, TIME:", diff)

    store_transform(transform)
