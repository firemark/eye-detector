#!/usr/bin/env python3
from argparse import ArgumentParser

from eye_detector.gen_to_train import utils
from eye_detector.gen_to_train.dump.eye import EyeDumper
from eye_detector.gen_to_train.data_loader import eye as data_loader
from eye_detector.model import store_transform


DATASETS = {
    'mrl': data_loader.MrlEyeDataLoader,
    'synth': data_loader.SynthEyeDataLoader,
    'helen': data_loader.HelenEyeDataLoader,
    'bioid': data_loader.BioIdEyeDataLoader,
}

parser = ArgumentParser(
    description="Generate data to learning detection of eye",
)
utils.fill_argparse(parser, DATASETS)
parser.add_argument('--face-multiplier', type=float, default=1.0)
parser.add_argument('--noise', type=float, default=0.0)


if __name__ == "__main__":
    args = parser.parse_args()
    data_loader = utils.get_img_data_loader(args, DATASETS, 'mrl')

    utils.recreate_directory("eye_to_train")
    transform = utils.find_transform(
        transform_name=args.transform,
        transform_image_name=args.image_transform,
    )
    dumper = EyeDumper(transform, args, data_loader)
    utils.dump_with_time(dumper)
    store_transform(transform, 'eye')
