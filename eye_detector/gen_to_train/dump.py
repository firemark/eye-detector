from os import mkdir
from shutil import rmtree
from contextlib import suppress

from joblib import Parallel, delayed, dump
from skimage.util import random_noise

from eye_detector.gen_to_train import data_loader
from eye_detector.const import EYE_LABEL, NOT_EYE_LABEL, FACE_LABEL


def recreate_directory():
    with suppress(OSError):
        rmtree("middata/eye_to_train")
    mkdir("middata/eye_to_train")


class Dumper():

    def __init__(self, transform_eye, config, eye_data_cls):
        self.transform_eye = transform_eye
        self.config = config
        self.eye_loader = eye_data_cls(self.config.chunk_size)
        self.room_loader = data_loader.RoomDataLoader()
        self.face_loader = data_loader.FaceDataLoader()

    def dump(self):
        chunks = self.eye_loader.chunk_it()
        generator = (
            delayed(self.task)(i, paths)
            for i, paths in enumerate(chunks)
        )
        Parallel(n_jobs=self.config.jobs)(generator)

    def task(self, i, paths):
        config = self.config
        print(f"{i * config.chunk_size: 6d}...")
        def transform(img):
            if config.noise > 1e-6:
                img = random_noise(img, var=config.noise)
            return self.transform_eye(img)
        load_image = self.eye_loader.load_image
        eyes = [
            transform(load_image(path))
            for path in paths
        ]

        if i == 0:
            dump(eyes[0].shape, "outdata/x_eye_shape")

        self.dump_to_file(eyes, EYE_LABEL, "eyes", i)
        count = len(eyes)
        del eyes

        def make_transform_and_dump(name, label, loader, multipler):
            if multipler < 1e-6:
                return
            data = [
                transform(image)
                for image in loader.load(count, multipler)
            ]
            self.dump_to_file(data, label, name, i)

        make_transform_and_dump(
            name="room",
            label=NOT_EYE_LABEL,
            loader=self.room_loader,
            multipler=config.room_multipler,
        )

        face_as_unique_label = config.face_as_unique_label
        face_label = FACE_LABEL if face_as_unique_label else NOT_EYE_LABEL
        make_transform_and_dump(
            name="face",
            label=face_label,
            loader=self.face_loader,
            multipler=config.face_multipler,
        )

    def dump_to_file(self, data, label, name, index):
        suffix = f"_{index:03d}_{name}"
        dump(data, f"middata/eye_to_train/x_{suffix}")
        dump([label] * len(data), f"middata/eye_to_train/y_{suffix}")
