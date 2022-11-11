from glob import glob
from pickle import load as pickle_load

from PIL import Image

from torch.utils.data import Dataset, random_split
from torch import FloatTensor
from torchvision import transforms

WIDTH = 60
HEIGHT = 40


def get_transform(with_resize=True):
    trans = [
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    if with_resize:
        trans.insert(0, transforms.Resize((WIDTH, HEIGHT)))
    return transforms.Compose(trans)


class GazeDataset(Dataset):

    def __init__(self, root: str):
        self.paths = glob(f"{root}/**/*.png", recursive=True)
        self.transform = get_transform()

    def __getitem__(self, index):
        image_filepath = self.paths[index]
        metadata_filepath = image_filepath.rpartition('.')[0] + '.pkl'

        with open(metadata_filepath, 'rb') as file:
            metadata = pickle_load(file)

        with open(image_filepath, 'rb') as file:
            img = self.transform(Image.open(file).convert('RGB'))

        gaze = FloatTensor(metadata['look_vec'])
        rot_matrix = FloatTensor(metadata['head_pose'])
        return {"rot_matrix": rot_matrix, "img": img}, gaze

    def __len__(self) -> int:
        return len(self.paths)


def create_dataset():
    dataset = GazeDataset(root="indata/SynthEyes_data")
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])
