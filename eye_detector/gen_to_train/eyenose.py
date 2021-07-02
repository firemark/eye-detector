#!/usr/bin/env python3
from argparse import ArgumentParser

from eye_detector.gen_to_train import utils
from eye_detector.gen_to_train.dump.eyenose import EyeNoseDumper
from eye_detector.gen_to_train.data_loader import eyenose as data_loader
from eye_detector.model import store_transform


DATASETS = {
    'helen': data_loader.HelenEyeNoseDataLoader,
    'bioid': data_loader.BioIdEyeNoseDataLoader,
}

parser = ArgumentParser(
    description="Generate data to learning detection of eye",
)
utils.fill_argparse(parser, DATASETS)
parser.add_argument('--face-multiplier', type=float, default=1.0)


if __name__ == "__main__":
    args = parser.parse_args()
    data_loader = utils.get_img_data_loader(args, DATASETS, 'bioid')

    utils.recreate_directory("eyenose_to_train")
    transform = utils.find_transform(
        transform_name=args.transform,
        transform_image_name=args.image_transform,
    )
    dumper = EyeNoseDumper(transform, args, data_loader)
    utils.dump_with_time(dumper)
    store_transform(transform, 'eyenose')
