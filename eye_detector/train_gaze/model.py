import torch.nn as nn

from .dataset import WIDTH, HEIGHT

KERNEL_SIZE = 5


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        fc1_w = ((WIDTH - KERNEL_SIZE + 1) // 2 - KERNEL_SIZE + 1) // 2
        fc1_h = ((HEIGHT - KERNEL_SIZE + 1) // 2 - KERNEL_SIZE + 1) // 2

        self.img_stack = nn.Sequential(
            nn.Conv2d(1, 6, KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, KERNEL_SIZE),
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

        self.final = nn.Bilinear(3, 60, 3)

    def forward(self, x):
        x_rot_matrix = self.rot_stack(x["rot_matrix"])
        x_img = self.img_stack(x["img"])
        return self.final(x_rot_matrix, x_img)
