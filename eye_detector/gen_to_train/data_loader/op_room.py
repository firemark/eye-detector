from glob import glob
from itertools import chain

from eye_detector.gen_to_train.data_loader.base import GenericLoader


class RoomDataLoader(GenericLoader):
    PATCH_SIZE = (128, 128)
    ROOM_GROUPS = [
        "buffet",
        "bedroom",
        "livingroom",
        "warehouse",
    ]

    def __init__(self):
        self.paths = list(chain.from_iterable(
            glob(f"indata/outdoor_images/{group}/*.jpg")
            for group in self.ROOM_GROUPS
        ))
