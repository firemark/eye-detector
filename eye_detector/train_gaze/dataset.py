from glob import glob
from itertools import chain
from pickle import load as pickle_load
from typing import List

from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import numpy as np

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


class SynthGazeDataset(Dataset):

    def __init__(self, root: str):
        paths = glob(f"{root}/**/*.png", recursive=True)
        self.cache = [
            self.load_image(filepath)
            for filepath in paths
        ]
        trans = [
            #transforms.RandomCrop(size=(76, 114)),
            #transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((WIDTH, HEIGHT)),
        ] + get_transform_components()
        self.transform = torch.jit.script(Sequential(*trans))

    def __getitem__(self, index):
        rot_matrix, img, gaze = self.cache[index]
        return (rot_matrix, self.transform(img)), gaze

    def load_image(self, image_filepath):
        metadata_filepath = image_filepath.rpartition('.')[0] + '.pkl'

        with open(metadata_filepath, 'rb') as file:
            metadata = pickle_load(file)

        with open(image_filepath, 'rb') as file:
            img = transforms.ToTensor()(Image.open(file).convert('rGR'))

        gaze = FloatTensor(metadata['look_vec'])
        rot_matrix = FloatTensor(metadata['head_pose'])
        return rot_matrix, img, gaze

    def __len__(self) -> int:
        return len(self.cache)


class MPIIIGazeDataset(Dataset):

    def __init__(self, root: str):
        annotation_paths = glob(f"{root}/Data/Original/p*/day*")
        self.face_model = loadmat(f"{root}/6 points-based face model.mat")['model'].transpose()
        self.cache = list(chain.from_iterable(
            self.load_annotation_and_images(filepath)
            for filepath in annotation_paths
        ))
        trans = [
            #transforms.RandomCrop(size=(76, 114)),
            #transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((WIDTH, HEIGHT)),
        ] + get_transform_components()
        self.transform = torch.jit.script(Sequential(*trans))

    def __getitem__(self, index):
        rot_matrix, img, gaze = self.cache[index]
        return (rot_matrix, self.transform(img)), gaze

    def load_annotation_and_images(self, dirpath):
        with open(f"{dirpath}/annotation.txt") as file:
            camera = loadmat(f"{dirpath}/../Calibration/Camera.mat")['cameraMatrix']
            for index, line in enumerate(file, start=1):
                if index % 4 != 0:
                    continue
                yield self.load_image(index, dirpath, annotation=line.split(), camera=camera)

    def load_image(self, index, dirpath, annotation, camera):
        image_filepath = f"{dirpath}/{index:04d}.jpg"
        with open(image_filepath, 'rb') as file:
            img = Image.open(file).convert('RGB')

        head_rot_vec = [float(x) for x in annotation[29:32]]
        head_tra_vec = [float(x) for x in annotation[32:35]]
        head_rot_mat = Rotation.from_rotvec(head_rot_vec)
        face_model_2d = head_rot_mat.apply(self.face_model) + head_tra_vec
        face_model_2d[:, 0] = face_model_2d[:, 0] * (camera[0, 0] / face_model_2d[:, 2]) + camera[0, 2]
        face_model_2d[:, 1] = face_model_2d[:, 1] * (camera[1, 1] / face_model_2d[:, 2]) + camera[1, 2]

        eye_left, eye_right = face_model_2d[2:4]
        eye_center = ((eye_left + eye_right) / 2.0).astype(int)
        size = abs(eye_left[0] - eye_right[0])
        w_size = int(size * 1.5)
        h_size = int(size)

        eye_img = img.crop((
            eye_center[0] - w_size // 2,
            eye_center[1] - h_size // 2,
            eye_center[0] + w_size // 2,
            eye_center[1] + h_size // 2,
        ))

        gaze_target = np.array([float(x) for x in annotation[26:29]])
        gaze = np.array([float(x) for x in annotation[35:38]]) - gaze_target
        gaze /= np.linalg.norm(gaze)

        tensor_img = transforms.ToTensor()(eye_img)
        tensor_head_rot_mat = FloatTensor(head_rot_mat.as_matrix().reshape(9))
        tensor_gaze = FloatTensor(gaze)
        return tensor_head_rot_mat, tensor_img, tensor_gaze

    def __len__(self) -> int:
        return len(self.cache)


def create_dataset():
    #dataset = SynthGazeDataset(root="indata/SynthEyes_data")
    dataset = MPIIIGazeDataset(root="indata/MPIIGaze")
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])
