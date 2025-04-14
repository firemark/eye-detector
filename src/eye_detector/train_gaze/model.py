import torch.nn as nn
from torch import Tensor, cat

from .dataset import WIDTH, HEIGHT

FIRST_KERNEL_SIZE = 5
SECOND_KERNEL_SIZE = 3


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_img_stack = self._create_img_stack()
        self.right_img_stack = self._create_img_stack()
        # self.rot_stack = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(9, 3),
        #     nn.ReLU(),
        # )
        self.final_stack = nn.Sequential(
            nn.Linear(3 + 256 * 2, 32),
            nn.ReLU(),
            # nn.Linear(256, 32),
            # nn.ReLU(),
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

        return nn.Sequential(
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