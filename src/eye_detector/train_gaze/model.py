import torch.nn as nn
from torch import Tensor, cat
from torchvision.models import vgg16

from .dataset import WIDTH, HEIGHT

FIRST_KERNEL_SIZE = 5
SECOND_KERNEL_SIZE = 3


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_img_stack, left_in_features = self._create_img_stack()
        self.right_img_stack, right_in_features = self._create_img_stack()
        # self.rot_stack = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(9, 3),
        #     nn.ReLU(),
        # )
        self.final_stack = nn.Sequential(
            nn.Linear(9 + left_in_features + right_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, x: tuple[Tensor, Tensor, Tensor]):
        # x_rot_matrix = self.rot_stack(x[0])
        x_left_img = self.left_img_stack(x[1])
        x_right_img = self.right_img_stack(x[2])
        x_final = cat((x[0], x_left_img, x_right_img), dim=1)
        return self.final_stack(x_final)

    def _create_img_stack(self):
        fc1_w = ((WIDTH - FIRST_KERNEL_SIZE + 1) // 2 - SECOND_KERNEL_SIZE + 1) // 2
        fc1_h = ((HEIGHT - FIRST_KERNEL_SIZE + 1) // 2 - SECOND_KERNEL_SIZE + 1) // 2
        features = nn.Sequential(
            nn.Conv2d(3, 6, FIRST_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, SECOND_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * fc1_w * fc1_h, 256),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
        )
        return features, 256

        model = vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        modules = [module for module in model.features]
        modules.append(model.avgpool)
        in_features = model.classifier[0].in_features
        features = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
        )

        for param in features.parameters():
            param.requires_grad = True

        return features, 1024

