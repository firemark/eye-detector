from glob import glob
from itertools import chain

from eye_detector.gen_to_train.data_loader.base import GenericLoader


class RoomDataLoader(GenericLoader):
    ROOM_GROUPS = [
        "buffet",
        "bedroom",
        "livingroom",
        "warehouse",
    ]

    def __init__(self):
        self.patch_size = (128, 128)
        self.paths = list(chain.from_iterable(
            glob(f"indata/outdoor_images/{group}/*.jpg")
            for group in self.ROOM_GROUPS
        ))
