import torch.nn as nn
import torch
from torch import Tensor, cat
from torchvision.models import vgg16, VGG16_Weights

from .dataset import WIDTH, HEIGHT

FIRST_KERNEL_SIZE = 5
SECOND_KERNEL_SIZE = 3


def create_new_net(device) -> "Net":
    net = Net().to(device).eval()
    example_input = [(torch.rand(1, 9), torch.rand(1, 3, WIDTH, HEIGHT), torch.rand(1, 3, WIDTH, HEIGHT))]
    example_input = [tuple(i.to(device) for i in example_input[0])]
    return torch.jit.trace(net, example_input)


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
            nn.Linear(9 + left_in_features + right_in_features, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 40),
            nn.ReLU(inplace=True),
            nn.Linear(40, 3),
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
            nn.Linear(16 * fc1_w * fc1_h, 80),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
        )
        return features, 80

        # model = vgg16(weights=VGG16_Weights.DEFAULT)

        # # remove the last ConvBRelu layer
        # modules = [module for module in model.features]
        # modules.append(model.avgpool)
        # in_features = model.classifier[0].in_features
        # features = nn.Sequential(
        #     *modules,
        #     nn.Flatten(),
        #     nn.Linear(in_features, 1024),
        #     nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
        #     nn.ReLU(inplace=True),
        # )

        # for param in features.parameters():
        #     param.requires_grad = True

        # return features, 1024
