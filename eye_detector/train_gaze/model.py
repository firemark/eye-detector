from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor

from .dataset import WIDTH, HEIGHT

FIRST_KERNEL_SIZE = 11
SECOND_KERNEL_SIZE = 5


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        fc1_w = ((WIDTH - FIRST_KERNEL_SIZE + 1) // 2 - SECOND_KERNEL_SIZE + 1) // 2
        fc1_h = ((HEIGHT - FIRST_KERNEL_SIZE + 1) // 2 - SECOND_KERNEL_SIZE + 1) // 2

        self.img_stack = nn.Sequential(
            nn.Conv2d(3, 6, FIRST_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, SECOND_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * fc1_w * fc1_h, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
        )

        self.rot_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9, 3),
            nn.ReLU(),
        )

        # self.final = nn.Bilinear(3, 60, 3)
        self.final_concat = nn.Bilinear(3, 60, 24)
        self.final_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(24, 3),
        )

    def forward(self, x: Tuple[Tensor, Tensor]):
        x_rot_matrix = self.rot_stack(x[0])
        x_img = self.img_stack(x[1])
        x_final = self.final_concat(x_rot_matrix, x_img)
        return self.final_stack(x_final)
        #return self.final(x_rot_matrix, x_img)
