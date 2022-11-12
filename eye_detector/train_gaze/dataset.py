from glob import glob
from pickle import load as pickle_load
from typing import List

from PIL import Image

from torch.utils.data import Dataset, random_split
from torch import FloatTensor
from torch.nn import Module, Sequential
from torchvision import transforms
import torch.jit

WIDTH = 60
HEIGHT = 40
KEY_ROT_MATRIX = 0
KEY_IMG = 1


def get_transform_components() -> List[Module]:
    return [
        transforms.Normalize([0.5], [0.5]),
    ]


def get_transform():
    return transforms.Compose([transforms.ToTensor()] + get_transform_components())


class GazeDataset(Dataset):

    def __init__(self, root: str):
        self.paths = glob(f"{root}/**/*.png", recursive=True)
        self.cache = {}
        trans = [
            #transforms.RandomCrop(size=(76, 114)),
            #transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((WIDTH, HEIGHT)),
        ] + get_transform_components()
        self.transform = torch.jit.script(Sequential(*trans))

    def __getitem__(self, index):
        obj = self.cache.get(index)
        if obj:
            rot_matrix, img, gaze = obj
            return ( rot_matrix, self.transform(img)), gaze

        image_filepath = self.paths[index]
        metadata_filepath = image_filepath.rpartition('.')[0] + '.pkl'

        with open(metadata_filepath, 'rb') as file:
            metadata = pickle_load(file)

        with open(image_filepath, 'rb') as file:
            img = transforms.ToTensor()(Image.open(file).convert('RGB'))

        gaze = FloatTensor(metadata['look_vec'])
        rot_matrix = FloatTensor(metadata['head_pose'])
        self.cache[index] = rot_matrix, img, gaze

        return (rot_matrix, self.transform(img)), gaze

    def __len__(self) -> int:
        return len(self.paths)


def create_dataset():
    dataset = GazeDataset(root="indata/SynthEyes_data")
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])
