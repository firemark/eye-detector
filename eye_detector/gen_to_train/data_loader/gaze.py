from glob import glob
from pickle import load as pickle_load

from skimage.transform import resize
from skimage import io

from eye_detector.gen_to_train.data_loader.base import ImgDataLoader


class SynthEyeDataLoader(ImgDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/SynthEyes_data/**/*.png", recursive=True)

    @staticmethod
    def load_image(filepath):
        metadata_filepath = filepath.rpartition('.')[0] + '.pkl'
        with open(metadata_filepath, 'rb') as file:
            metadata = pickle_load(file)

        gaze = metadata['look_vec']
        rot_matrix = metadata['head_pose']
        img = io.imread(filepath)[:, :, 0:3]
        return gaze, rot_matrix, resize(img, [60, 40])
