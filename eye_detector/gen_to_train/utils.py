from os import mkdir
from shutil import rmtree
from contextlib import suppress
from time import time

from eye_detector.gen_to_train.data_loader import MultiImgDataLoader
from eye_detector.gen_to_train.transforms import TRANSFORMS
from eye_detector.gen_to_train.image_transforms import IMAGE_TRANSFORMS


def fill_argparse(parser, datasets):
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('-j', '--jobs', type=int, default=8)
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
    parser.add_argument(
        '-d', '--dataset',
        dest='datasets',
        nargs='+',
        choices=list(datasets.keys()),
    )


def recreate_directory(name):
    with suppress(OSError):
        rmtree(f"middata/{name}")
    mkdir(f"middata/{name}")


def get_img_data_loader(args, datasets_cls, default_dataset):
    datasets = args.datasets or [default_dataset]
    eye_classes = [datasets_cls[cls] for cls in datasets]

    if len(eye_classes) == 1:
        eye_data_cls = eye_classes[0]
        eye_data_cls.assert_args(args)
        return eye_data_cls(args.chunk_size)

    for cls in eye_classes:
        cls.assert_args(args)

    return MultiImgDataLoader(
        chunk_size=args.chunk_size,
        eye_loaders=[cls(args.chunk_size) for cls in eye_classes],
    )


def find_transform(transform_name, transform_image_name):
    image_transform = IMAGE_TRANSFORMS[transform_image_name]
    transform = TRANSFORMS[transform_name]
    transform.set_image_transform(image_transform)

    return transform


def dump_with_time(dumper):
    t = time()
    dumper.dump()
    diff = time() - t
    print("GENERATION DONE, TIME:", diff)
