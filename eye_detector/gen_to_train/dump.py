from os import mkdir
from shutil import rmtree
from contextlib import suppress

from joblib import Parallel, delayed, dump

from eye_detector.gen_to_train import data_loader


def recreate_directory():
    with suppress(OSError):
        rmtree("middata/eye_to_train")
    mkdir("middata/eye_to_train")


class Dumper():
    EYE_LABEL = 1
    ROOM_LABEL = 0
    FACE_LABEL = 2
    LABELS = (EYE_LABEL, FACE_LABEL, ROOM_LABEL)

    def __init__(self, transform_eye, config):
        self.transform_eye = transform_eye
        self.config = config
        self.eye_loader = data_loader.EyeDataLoader(self.config.chunk_size)
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
        transform = self.transform_eye
        load_image = self.eye_loader.load_image
        eyes = [transform(load_image(path)) for path in paths]

        if i == 0:
            print("shape:", eyes[0].shape)
            dump(eyes[0].shape, "outdata/x_eye_shape")

        self.dump_to_file(eyes, self.EYE_LABEL, "eyes", i)
        count = len(eyes)
        del eyes

        def make_transform_and_dump(name, label, loader, multipler):
            data = [
                transform(image) for image
                in loader.load(count, multipler)
            ]
            self.dump_to_file(data, label, name, i)

        make_transform_and_dump(
            name="room",
            label=self.ROOM_LABEL,
            loader=self.room_loader,
            multipler=config.room_multipler,
        )

        make_transform_and_dump(
            name="face",
            label=self.FACE_LABEL,
            loader=self.face_loader,
            multipler=config.face_multipler,
        )

    def dump_to_file(self, data, label, name, index):
        suffix = f"_{index:03d}_{name}"
        dump(data, f"middata/eye_to_train/x_{suffix}")
        dump([label] * len(data), f"middata/eye_to_train/y_{suffix}")
