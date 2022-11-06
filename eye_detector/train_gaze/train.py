from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader


def train(dataset, net, max_epoch=10):
    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    criterion = MSELoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(max_epoch):  # loop over the dataset multiple times
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

            # print statistics
            epoch_running_loss += loss.item()
            running_loss += loss.item()
            total += len(gazes)
            if i % 500 == 0:
                print('[%2d, %5d] loss: %.5f' %
                      (epoch + 1, total, running_loss / 2000))
                running_loss = 0.0
        print('[%2d, %5d] epoch loss: %.5f' %
              (epoch + 1, total, epoch_running_loss / 2000))
