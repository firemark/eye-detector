from time import monotonic

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


LOG_FORMAT = '[%2d, %7d / %7d] loss: %.5f; time: %.3f'
EPS = 1e-3


def angle_loss(output, target):
    output_norm = torch.norm(output, dim=1)
    target_norm = torch.norm(target, dim=1)
    dot = (output * target).sum(axis=1)
    return (dot / (output_norm * target_norm)).clamp(-1+EPS, +1-EPS).acos().sum()


def train(dataset_train, dataset_test, test, net, max_epoch=10):
    trainloader = DataLoader(dataset_train, batch_size=20, shuffle=True, num_workers=4)
    data_size = len(trainloader)
    # criterion = MSELoss()
    criterion = angle_loss
    optimizer = Adam(net.parameters(), lr=0.001)

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        epoch_time = monotonic()
        time = monotonic()
        epoch_running_loss = 0.0
        running_loss = 0.0
        total = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, gazes = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, gazes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_running_loss += loss.item()
            total += len(gazes)
            if i != 0 and i % 200 == 0:
                print(LOG_FORMAT % (epoch + 1, total, data_size, running_loss / 500, monotonic() - time))
                running_loss = 0.0
                time = monotonic()

        print('Epoch:', LOG_FORMAT % (epoch + 1, total, data_size, epoch_running_loss / total, monotonic() - epoch_time))
        if (epoch < max_epoch - 5 and epoch % 5 == 4) or epoch == max_epoch - 1:
            print('---')
            print('test result:')
            test(dataset_test, net)
            print('---')
