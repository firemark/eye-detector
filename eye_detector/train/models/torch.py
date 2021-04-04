import numpy as np
from torch import as_tensor, no_grad
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NET_W = 32
NET_H = 32
KERNEL_SIZE = 5


class AsModelNet(nn.Module):
    shape = (1, NET_W, NET_H)

    @staticmethod
    def chunk_it(xx):
        d = []
        for x in xx:
            d.append(x)
            if len(d) >= 3:
                yield as_tensor(d)
                d = []
        if d:
            yield as_tensor(d)

    def fit(self, xx, yy):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):
            chunk_x = (
                as_tensor(
                    x
                    .reshape(1, *self.shape)
                    .astype(np.float32)
                )
                for x in xx
            )
            chunk_y = (
                as_tensor(y.reshape(1))
                for y in yy
            )
            for x, y in zip(chunk_x, chunk_y):
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def predict(self, xx):
        with no_grad():
            tensor = self(
                as_tensor(
                    xx
                    .reshape(-1, *self.shape)
                    .astype(np.float32)
                )
            )
            return (
                tensor
                .numpy()
                .argmax(axis=1)
            )

    def eye_probability(self, vector):
        vector = vector.reshape(1, -1)
        with no_grad():
            tensor = self(
                as_tensor(
                    vector
                    .reshape(-1, *self.shape)
                    .astype(np.float32)
                )
            )
        return tensor[0, 1].item()


class Net(AsModelNet):

    def __init__(self):
        super().__init__()
        self.fc1_w = (NET_W - KERNEL_SIZE + 1) // 2
        self.fc1_h = (NET_H - KERNEL_SIZE + 1) // 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(1, 12, KERNEL_SIZE)
        self.fc1 = nn.Linear(12 * self.fc1_w * self.fc1_h, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        #x = x.view(3, 1, NET_W, NET_H)
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 12 * self.fc1_w * self.fc1_h)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class Net2(AsModelNet):

    def __init__(self, shape_size):
        super().__init__()
        mid_size = round(np.sqrt(shape_size))
        self.shape = (shape_size,)
        self.fc1 = nn.Linear(shape_size, mid_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(mid_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def torch(x, y, shape):
    if len(shape) == 2:
        return Net()
    else:
        return Net2(np.prod(shape))
