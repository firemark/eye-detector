from joblib import Parallel, delayed, dump
from skimage.util import random_noise

from eye_detector.gen_to_train import data_loader
from eye_detector.const import EYE_LABEL, NOT_EYE_LABEL, FACE_LABEL


class Dumper():
    LOADER_NAME = 'img'

    def __init__(self, transform_img, config, loader):
        self.transform_img = transform_img
        self.config = config
        self.loader = loader
        self.another_loaders = []

    def dump(self):
        chunks = self.loader.chunk_it()
        generator = (
            delayed(self.task)(i, paths)
            for i, paths in enumerate(chunks)
        )
        Parallel(n_jobs=self.config.jobs)(generator)

    def task(self, i, paths):
        config = self.config
        print(f"{i * config.chunk_size: 6d}...")
        load_image = self.loader.load_image
        imgs = [
            self.transform(img)
            for img in (load_image(path) for path in paths)
            if img is not None
        ]

        if i == 0:
            dump(imgs[0].shape, f"outdata/x_{self.LOADER_NAME}_shape")

        self.dump_to_file(imgs, EYE_LABEL, self.LOADER_NAME, i)
        count = len(imgs)
        del imgs

        for loader_info in self.another_loaders:
            self.make_transform_and_dump(**loader_info, count=count, i=i)

    def transform(self, img):
        return self.transform_img(img)

    def make_transform_and_dump(self, name, label, loader, multipler, count, i):
        if multipler < 1e-6:
            return
        data = [
            self.transform(image)
            for image in loader.load(count, multipler)
        ]
        self.dump_to_file(data, label, name, i)

    def dump_to_file(self, data, label, name, index):
        name = f"{index:03d}_{name}"
        dump(
            {
                "x": data,
                "y": [label] * len(data),
            },
            f"middata/{self.LOADER_NAME}_to_train/{name}"
        )


class EyeDumper(Dumper):
    LOADER_NAME = 'eye'

    def __init__(self, transform_img, config, loader):
        super().__init__(transform_img, config, loader)
        self.another_loaders = [
            dict(
                name='face',
                label=NOT_EYE_LABEL,
                loader=data_loader.FaceDataLoader(),
                multiplier=config.face_multiplier,
            ),
        ]

    def transform(self):
        noise = self.config.noise * 0.01
        if noise > 1e-6:
            img = random_noise(img, var=noise)
        return self.transform_img(img)


class FaceDumper(Dumper):
    LOADER_NAME = 'face'

    def __init__(self, transform_img, config, loader):
        super().__init__(transform_img, config, loader)
        self.another_loaders = [
            dict(
                name='room',
                label=NOT_EYE_LABEL,
                loader=data_loader.RoomDataLoader(),
                multipler=config.room_multiplier,
            ),
        ]
