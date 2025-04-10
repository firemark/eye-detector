from eye_detector.gen_to_train.dump.base import Dumper
from eye_detector.gen_to_train.data_loader.op_room import RoomDataLoader
from eye_detector.const import NOT_EYE_LABEL


class FaceDumper(Dumper):
    LOADER_NAME = 'face'

    def __init__(self, transform_img, config, loader):
        super().__init__(transform_img, config, loader)
        self.another_loaders = [
            dict(
                name='room',
                label=NOT_EYE_LABEL,
                loader=RoomDataLoader(),
                multiplier=config.room_multiplier,
            ),
        ]
