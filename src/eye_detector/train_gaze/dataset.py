from glob import glob
from pickle import load as pickle_load
from typing import List, Literal

from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import numpy as np

from torch.utils.data import Dataset, ConcatDataset, random_split
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
            # transforms.RandomCrop(size=0.95),
            transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
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
            img = transforms.ToTensor()(Image.open(file).convert('RGB'))

        gaze = FloatTensor(metadata['look_vec'])
        rot_matrix = FloatTensor(metadata['head_pose'])
        return rot_matrix, img, gaze

    def __len__(self) -> int:
        return len(self.cache)


class MPIIIGazeDataset(Dataset):

    def __init__(self, root: str, eye: Literal["left", "right"]):
        annotation_paths = glob(f"{root}/Data/Original/p*/day*")
        self.face_model = loadmat(f"{root}/6 points-based face model.mat")['model'].transpose()
        self.paths = list(x for i, x in enumerate(self.get_paths(annotation_paths)) if i % 10 == 0)
        self.camera_cache = {}
        self.cache = {}
        self.eye = eye
        trans = [
            # transforms.RandomCrop(size=0.95),
            transforms.ColorJitter(hue=0.02, saturation=0.02, contrast=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((WIDTH, HEIGHT)),
        ] + get_transform_components()
        self.transform = torch.jit.script(Sequential(*trans))

    def __getitem__(self, index):
        obj = self.cache.get(index)
        if obj is None:
            dirpath, annotation, img_path = self.paths[index]
            obj = self.load_image(dirpath, annotation, img_path)
            self.cache[index] = obj

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
        head_rot_vec = [float(x) for x in annotation[29:32]]
        head_tra_vec = [float(x) for x in annotation[32:35]]
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

        tensor_left_img = transforms.ToTensor()(left_eye_img)
        tensor_right_img = transforms.ToTensor()(right_eye_img)

        tensor_head_rot_mat = FloatTensor(head_rot_mat.as_matrix())
        tensor_gaze = FloatTensor(gaze)
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


def create_dataset(eye):
    # dataset = SynthGazeDataset(root="indata/SynthEyes_data")
    dataset = MPIIIGazeDataset(root="indata/MPIIGaze", eye=eye)
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])
