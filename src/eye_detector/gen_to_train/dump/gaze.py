from joblib import dump
from numpy import array

from eye_detector.gen_to_train.dump.base import DumperBase


class GazeDumper(DumperBase):
    LOADER_NAME = "gaze"

    def transform(self, obj):
        gaze, rot_matrix, img = obj
        return array(gaze), array(rot_matrix), self.transform_img(img)

    def task(self, i, paths):
        objs = self.get_chunked_objs(i, paths)

        if i == 0:
            dump(objs[0][-1].shape, f"outdata/{self.LOADER_NAME}_shape")

        self.dump_to_file(objs, i)

    def dump_to_file(self, data, index):
        dump(
            {
                "x": [(rot_matrix, img) for gaze, rot_matrix, img in data],
                "y": [gaze for gaze, rot_matrix, img in data],
            },
            f"middata/{self.LOADER_NAME}_to_train/{index:03d}"
        )
