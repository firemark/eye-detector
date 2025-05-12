from glob import glob
from pickle import load as pickle_load
from random import shuffle
from typing import List
from uuid import uuid4
from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import numpy as np

from torch.utils.data import Dataset
from torch import FloatTensor
from torch.nn import Module, Sequential
from torchvision import transforms
import torch.jit

WIDTH = 30
HEIGHT = 18


def get_transform_components() -> List[Module]:
    return [
        #transforms.Normalize([0.5], [0.5]),
    ]


def get_transform():
    return transforms.Compose([transforms.ToTensor()] + get_transform_components())


class SynthGazeDataset(Dataset):

    def __init__(self, device, root: str, size_ratio=1.0):
        self.paths = glob(f"{root}/**/*.png", recursive=True)
        shuffle(self.paths)
        if size_ratio < 1.0:
            self.paths = self.paths[:int(len(self.paths) * size_ratio)]
        trans = [
            # transforms.RandomCrop(size=0.95),
            transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((HEIGHT, WIDTH)),
        ] + get_transform_components()
        # selfransform = torch.jit.script(Sequential(*trans)).to(device)
        self.transform = Sequential(*trans).to(device)
        self.device = device

    def __getitem__(self, index):
        path = self.paths[index]
        rot_matrix, left, right, gaze = self.load_image(path)
        return (rot_matrix, self.transform(left), self.transform(right)), gaze

    def load_image(self, image_filepath):
        metadata_filepath = image_filepath.rpartition('.')[0] + '.pkl'

        with open(metadata_filepath, 'rb') as file:
            metadata = pickle_load(file)

        with open(image_filepath, 'rb') as file:
            img = transforms.ToTensor()(Image.open(file).convert('RGB')).to(self.device)

        x, y, z = metadata['look_vec']
        gaze = FloatTensor([-z, x, y]).to(self.device)
        rot_vec = FloatTensor(metadata['head_pose']).reshape(9).to(self.device)
        return rot_vec, img, img, gaze

    def __len__(self) -> int:
        return len(self.paths)


class MPIIIGazeDataset(Dataset):

    def __init__(self, device, root: str, size_ratio: float = 1.0):
        annotation_paths = glob(f"{root}/Data/Original/p*/day*")
        self.face_model = loadmat(f"{root}/6 points-based face model.mat")['model'].transpose()
        self.paths = list(self.get_paths(annotation_paths))
        shuffle(self.paths)
        if size_ratio < 1.0:
            self.paths = self.paths[:int(len(self.paths) * size_ratio)]
        self.camera_cache = {}
        self.cache = {}
        trans = [
            # transforms.RandomCrop(size=0.95),
            transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((HEIGHT, WIDTH)),
        ] + get_transform_components()
        # self.transform = torch.jit.script(Sequential(*trans)).to(device)
        self.transform = Sequential(*trans).to(device)
        self.device = device

    def __getitem__(self, index):
        dirpath, annotation, img_path = self.paths[index]
        obj = self.load_image(dirpath, annotation, img_path)
        rot_matrix, left_img, right_img, gaze = obj
        return (rot_matrix, self.transform(left_img), self.transform(right_img)), gaze

    def __len__(self) -> int:
        return len(self.paths)

    def get_paths(self, dirpaths):
        for dirpath in dirpaths:
            with open(f"{dirpath}/annotation.txt") as file:
                for index, line in enumerate(file, start=1):
                    yield dirpath, line.split(), f"{dirpath}/{index:04d}.jpg"

    def load_camera(self, dirpath):
        camera_path = f"{dirpath}/../Calibration/Camera.mat"
        camera = self.camera_cache.get(camera_path)
        if camera is None:
            camera = loadmat(camera_path)['cameraMatrix']
            self.camera_cache[camera_path] = camera
        return camera

    def load_image(self, dirpath, annotation, image_filepath):
        with open(image_filepath, 'rb') as file:
            img = Image.open(file).convert('RGB')

        camera = self.load_camera(dirpath)
        head_rot_vec = np.array([float(x) for x in annotation[29:32]])
        head_tra_vec = np.array([float(x) for x in annotation[32:35]])
        head_rot_mat = Rotation.from_rotvec(head_rot_vec)
        face_model_2d = head_rot_mat.apply(self.face_model) + head_tra_vec
        face_model_2d[:, 0] = face_model_2d[:, 0] * (camera[0, 0] / face_model_2d[:, 2]) + camera[0, 2]
        face_model_2d[:, 1] = face_model_2d[:, 1] * (camera[1, 1] / face_model_2d[:, 2]) + camera[1, 2]

        left_eye_img = self.__get_eye_img(img, face_model_2d[0], face_model_2d[1])
        right_eye_img = self.__get_eye_img(img, face_model_2d[2], face_model_2d[3])
        left_eye_position = np.array([float(x) for x in annotation[35:38]])
        right_eye_position = np.array([float(x) for x in annotation[38:41]])
        position = (left_eye_position + right_eye_position) / 2

        gaze_target = np.array([float(x) for x in annotation[26:29]])
        gaze = position - gaze_target
        gaze /= np.linalg.norm(gaze)

        tensor_left_img = transforms.ToTensor()(left_eye_img).to(self.device)
        tensor_right_img = transforms.ToTensor()(right_eye_img).to(self.device)

        nx, ny, nz = head_rot_vec
        x, y, z = gaze
        rot_matrix = Rotation.from_rotvec([nz, nx, -ny]).as_matrix()
        tensor_head_rot_mat = FloatTensor(rot_matrix.reshape(9)).to(self.device)
        tensor_gaze = FloatTensor([z, x, -y]).to(self.device)
        return tensor_head_rot_mat, tensor_left_img, tensor_right_img, tensor_gaze

    def __get_eye_img(self, img, left, right):
        center = ((left + right) / 2.0).astype(int)
        size = abs(left[0] - right[0])
        w_size = int(size * 1.5)
        h_size = int(size)
        return img.crop((
            center[0] - w_size // 2,
            center[1] - h_size // 2,
            center[0] + w_size // 2,
            center[1] + h_size // 2,
        ))


def create_dataset(device, size_ratio=1.0):
    # dataset = SynthGazeDataset(root="indata/SynthEyes_data")
    train_dataset = MPIIIGazeDataset(device, root="indata/MPIIGaze", size_ratio=size_ratio)
    # test_dataset = SynthGazeDataset(device, root="indata/SynthEyes_data", size_ratio=0.5)

    train_size = int(len(train_dataset) * 0.8)
    test_size = len(train_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    return train_dataset, test_dataset
