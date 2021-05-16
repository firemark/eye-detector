import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split

from eye_detector.const import CLASSES

NET_W = 128
NET_H = 64
KERNEL_SIZE = 5

transform = transforms.Compose([
    transforms.Resize((NET_W, NET_H)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_w = ((NET_W - KERNEL_SIZE + 1) // 2 - KERNEL_SIZE + 1) // 2
        self.fc1_h = ((NET_H - KERNEL_SIZE + 1) // 2 - KERNEL_SIZE + 1) // 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, KERNEL_SIZE)
        self.conv2 = nn.Conv2d(6, 16, KERNEL_SIZE)
        self.fc1 = nn.Linear(16 * self.fc1_w * self.fc1_h, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.fc1_w * self.fc1_h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataset, net):
    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test_data(dataset, net):
    testloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    class_correct = [0.0] * len(CLASSES)
    class_total = [0.0] * len(CLASSES)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels)
            for i, label in enumerate(labels):
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    for i, klass in enumerate(CLASSES):
        ratio = class_correct[i] / class_total[i]
        print(f"Accuracy of {klass:10s} : {100 * ratio:06.2f}%")


def create_dataset():
    dataset = ImageFolder(root='middata/transformed_label', transform=transform)
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])


if __name__ == "__main__":
    net = Net()
    trainset, testset = create_dataset()

    train(trainset, net)
    test_data(testset, net)
    torch.save(net.state_dict(), "outdata/net.pth")
    print('Finished Training')
