import os
from random import randint
from glob import glob

from skimage.transform import resize
from skimage.color import gray2rgb
from skimage import io

from eye_detector.gen_to_train.data_loader.base import ImgDataLoader


class MrlEyeDataLoader(ImgDataLoader):
    EYE_STATE_COL = 4
    LIGHT_CONDITION_COL = 6

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/mrlEyes_2018_01/**/*.png", recursive=True)

    @classmethod
    def assert_args(self, args):
        if args.image_transform != 'gray':
            raise AssertionError("MRL dataset support only gray images")

    def load(self):
        for path in super().load():
            filename = os.path.basename(path).partition(".")[0]
            features = filename.split("_")
            if features[self.EYE_STATE_COL] != "1":
                continue
            if features[self.LIGHT_CONDITION_COL] != "1":
                continue
            yield path

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)
        return resize(gray2rgb(img), [32, 64])


class BioIdEyeDataLoader(ImgDataLoader):
    HEIGHT = 286
    WIDTH = 384

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/bioid/*.pgm", recursive=True)

    @classmethod
    def assert_args(self, args):
        if args.image_transform != 'gray':
            raise AssertionError("BioId dataset support only gray images")

    def load(self):
        for path in super().load():
            dirname = os.path.dirname(path)
            filename = os.path.basename(path)
            without_suffix = filename.partition('.')[0]
            eye_info_path = os.path.join(dirname, without_suffix + ".eye")
            with open(eye_info_path) as file:
                line = file.readline() # skip first line
                assert line == "#LX	LY	RX	RY\n"
                eye_info = file.readline().split()
            left_cord = eye_info[0:2]
            right_cord = eye_info[2:4]
            for cord in [left_cord, right_cord]:
                for i in range(3):
                    x, y = cord
                    x = int(x) + randint(0, 3)
                    y = int(y) + randint(0, 3)
                    bbox = [
                        max(x - 16, 0),
                        min(x + 16, self.WIDTH - 1),
                        max(y - 8, 0),
                        min(y + 8, self.HEIGHT - 1),
                    ]
                    yield path, bbox

    @classmethod
    def load_image(cls, data):
        filepath, bbox = data
        (x1, x2, y1, y2) = bbox
        img = io.imread(filepath)
        img = img[y1:y2, x1:x2]
        return resize(gray2rgb(img), [32, 64])


class SynthEyeDataLoader(ImgDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/SynthEyes_data/**/*.png", recursive=True)

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)[:, :, 0:3]
        return resize(img, [32, 64])


class HelenEyeDataLoader(ImgDataLoader):

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
                start = 41 + 17 + 28 * 2
                r_eye_raw = lines[start:start + 20]
                l_eye_raw = lines[start + 20:start + 40]

            filepath = filepaths[0]
            for eye in [r_eye_raw, l_eye_raw]:
                for i in range(3):
                    bbox = self.get_bbox(eye)
                    if bbox:
                        yield filepath, bbox

    @staticmethod
    def get_bbox(raw):
        gen = (o.partition(',') for o in raw)
        xy = [(float(x), float(y)) for x, _, y in gen]
        min_x = round(min(x for x, y in xy)) - 5
        max_x = round(max(x for x, y in xy)) + 5
        min_y = round(min(y for x, y in xy)) - 5
        max_y = round(max(y for x, y in xy)) + 5
        dx = max_x - min_x
        dy = max_y - min_y
        dx1 = dx // 5
        dy1 = dy // 5
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
        return resize(img, [32, 64])
