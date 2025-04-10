#!/usr/bin/env python3
from argparse import ArgumentParser

from eye_detector.gen_to_train import utils
from eye_detector.gen_to_train.dump.gaze import GazeDumper
from eye_detector.gen_to_train.data_loader import gaze as data_loader
from eye_detector.model import store_transform

DATASETS = {
    'synth': data_loader.SynthEyeDataLoader,
}

parser = ArgumentParser(
    description="Generate data to learning detection of gaze",
)
utils.fill_argparse(parser, DATASETS)


if __name__ == "__main__":
    args = parser.parse_args()
    data_loader = utils.get_img_data_loader(args, DATASETS, 'synth')

    utils.recreate_directory("gaze_to_train")
    transform = utils.find_transform(
        transform_name=args.transform,
        transform_image_name=args.image_transform,
    )
    dumper = GazeDumper(transform, args, data_loader)
    utils.dump_with_time(dumper)
    store_transform(transform, 'face')
