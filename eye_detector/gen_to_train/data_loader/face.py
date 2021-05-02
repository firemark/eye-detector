import os
from glob import glob
from random import randint

from skimage.transform import resize
from skimage import io

from eye_detector.gen_to_train.data_loader.base import ImgDataLoader


class HelenFaceDataLoader(ImgDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = list(self.get_paths_from_annotations())

    def get_paths_from_annotations(self):
        print("GENERATING PATHS TO IMAGES")
        annotation_paths = glob("indata/helen/annotation/**/*.txt", recursive=True)
        for path in annotation_paths:
            with open(path) as file:
                filename = file.readline().strip()
                seekpath = f"indata/helen/**/{filename}.jpg"
                filepaths = glob(seekpath, recursive=True)
                if len(filepaths) == 0:
                    # probably a test file. Skip.
                    continue
                lines = file.readlines()

                # FUT standard
                start = 0
                end = 41 + 17 + 28 * 2 + 20 * 2
                points = lines[start:end]

            filepath = filepaths[0]
            for i in range(3):
                bbox = self.get_bbox(points)
                if bbox:
                    yield filepath, bbox

    @staticmethod
    def get_bbox(raw):
        gen = (o.partition(',') for o in raw)
        xy = [(float(x), float(y)) for x, _, y in gen]
        min_x = round(min(x for x, y in xy))
        max_x = round(max(x for x, y in xy))
        min_y = round(min(y for x, y in xy))
        max_y = round(max(y for x, y in xy))
        dx = max_x - min_x
        dy = max_y - min_y
        dx1 = dx // 10
        dy1 = dy // 10
        cx = min_x + dx // 2 + randint(-dx1, dx1)
        cy = min_y + dy // 2 + randint(-dy1, dy1)
        dh = max(dx, dy) // 2

        if cx - dh < 0 or cy - dh < 0:
            return None

        return (cx - dh, cx + dh, cy - dh, cy + dh)

    @staticmethod
    def load_image(data):
        filepath, bbox = data
        (x1, x2, y1, y2) = bbox
        img = io.imread(filepath)
        img = img[y1:y2, x1:x2]
        return resize(img, [128, 128])
