from skimage.util import random_noise

from eye_detector.gen_to_train.dump.base import Dumper
from eye_detector.gen_to_train.data_loader.op_face import FaceDataLoader
from eye_detector.const import NOT_EYE_LABEL


class EyeNoseDumper(Dumper):
    LOADER_NAME = 'eye'

    def __init__(self, transform_img, config, loader):
        super().__init__(transform_img, config, loader)
        self.another_loaders = [
            dict(
                name='face',
                label=NOT_EYE_LABEL,
                loader=FaceDataLoader(cols=3),
                multiplier=config.face_multiplier,
            ),
        ]
