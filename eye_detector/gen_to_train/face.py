#!/usr/bin/env python3
from argparse import ArgumentParser

from eye_detector.gen_to_train import utils
from eye_detector.gen_to_train.dump.face import FaceDumper
from eye_detector.gen_to_train.data_loader import face as data_loader
from eye_detector.model import store_transform

DATASETS = {
    'helen': data_loader.HelenFaceDataLoader,
}

parser = ArgumentParser(
    description="Generate data to learning detection of face",
)
utils.fill_argparse(parser, DATASETS)
parser.add_argument('--room-multiplier', type=float, default=1.0)


if __name__ == "__main__":
    args = parser.parse_args()
    data_loader = utils.get_img_data_loader(args, DATASETS, 'helen')

    utils.recreate_directory("face_to_train")
    transform = utils.find_transform(
        transform_name=args.transform,
        transform_image_name=args.image_transform,
    )
    dumper = FaceDumper(transform, args, data_loader)
    utils.dump_with_time(dumper)
    store_transform(transform, 'face')
