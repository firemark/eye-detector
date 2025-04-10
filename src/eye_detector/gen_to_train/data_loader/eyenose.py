import os
from random import randint
from glob import glob

from skimage.transform import resize
from skimage.color import gray2rgb
from skimage import io

from eye_detector.gen_to_train.data_loader.base import ImgDataLoader


class BioIdEyeNoseDataLoader(ImgDataLoader):
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
            for i in range(3):
                left_x, left_y = left_cord
                right_x, right_y = right_cord
                left_x = int(left_x) + randint(0, 10)
                left_y = int(left_y) + randint(0, 5)
                right_x = int(right_x) + randint(0, 10)
                right_y = int(right_y) + randint(0, 5)
                bbox = [
                    max(right_x - 16, 0),
                    min(left_x + 16, self.WIDTH - 1),
                    max(min(left_y, left_y) - 16, 0),
                    min(max(right_y, right_y) + 16, self.HEIGHT - 1),
                ]
                yield path, bbox

    @classmethod
    def load_image(cls, data):
        filepath, bbox = data
        x1, x2, y1, y2 = bbox
        img = io.imread(filepath)
        img = img[y1:y2, x1:x2]
        return resize(gray2rgb(img), [64, 64 * 3])


class HelenEyeNoseDataLoader(ImgDataLoader):

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
            for i in range(3):
                l_bbox = self.get_bbox(l_eye_raw)
                r_bbox = self.get_bbox(r_eye_raw)

                if not all([l_bbox, r_bbox]):
                    continue

                left_x, _, left_down_y, left_up_y = l_bbox
                _, right_x, right_down_y, right_up_y = r_bbox
                down_y = min(left_down_y, right_down_y)
                up_y = max(left_up_y, right_up_y)
                if up_y < down_y:
                    up_y, down_y = down_y, up_y
                bbox = [left_x, right_x, down_y, up_y]

                yield filepath, bbox


    @staticmethod
    def get_bbox(raw):
        gen = (o.partition(',') for o in raw)
        xy = [(float(x), float(y)) for x, _, y in gen]
        min_x = round(min(x for x, y in xy)) - 35
        max_x = round(max(x for x, y in xy)) + 35
        min_y = round(min(y for x, y in xy)) - 25
        max_y = round(max(y for x, y in xy)) + 25
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
        return resize(img, [64, 64 * 3])
