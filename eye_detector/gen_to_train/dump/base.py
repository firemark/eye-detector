from joblib import Parallel, delayed, dump

from eye_detector.const import EYE_LABEL

class DumperBase:
    def __init__(self, transform_img, config, loader):
        self.transform_img = transform_img
        self.config = config
        self.loader = loader

    def transform(self, img):
        return self.transform_img(img)

    def dump(self):
        chunks = self.loader.chunk_it()
        generator = (
            delayed(self.task_wrapped)(i, paths)
            for i, paths in enumerate(chunks)
        )
        Parallel(n_jobs=self.config.jobs)(generator)

    def get_chunked_objs(self, i, paths):
        load_image = self.loader.load_image
        return [
            self.transform(img)
            for img in (load_image(path) for path in paths)
            if img is not None
        ]

    def task_wrapped(self, i, paths):
        config = self.config
        print(f"{i * config.chunk_size: 6d}...")
        self.task(i, paths)

    def task(self, i, paths):
        pass


class Dumper(DumperBase):
    LOADER_NAME = 'img'

    def __init__(self, transform_img, config, loader):
        super().__init__(transform_img, config, loader)
        self.another_loaders = []

    def task(self, i, paths):
        super().task(i, paths)
        imgs = self.get_chunked_objs(paths)

        if i == 0:
            dump(imgs[0].shape, f"outdata/x_{self.LOADER_NAME}_shape")

        self.dump_to_file(imgs, EYE_LABEL, self.LOADER_NAME, i)
        count = len(imgs)
        del imgs

        for loader_info in self.another_loaders:
            self.make_transform_and_dump(**loader_info, count=count, i=i)

    def transform(self, img):
        return self.transform_img(img)

    def make_transform_and_dump(self, name, label, loader, multiplier, count, i):
        if multiplier < 1e-6:
            return

        data = [
            self.transform(image)
            for image in loader.load(count, multiplier)
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
