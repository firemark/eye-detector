from time import monotonic
import torch
from torch.utils.data import DataLoader


def test_data(dataset, net):
    testloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    err = 0.0
    time = monotonic()


    with torch.no_grad():
        for data in testloader:
            images, gazes = data
            outputs = net(images)
            err += torch.sum(torch.abs(gazes - outputs)).item()

    print('     total:', len(dataset))
    print('mean error:', err / len(dataset))
    print('      time:', monotonic() - time)
    # print(' '.join('%12s' % label for label in ('-',) + CLASSES))
    # for y, label in enumerate(CLASSES):
    #     print(
    #         '%12s' % label,
    #         ' '.join('%12d' % matrix[x, y] for x in range(len(CLASSES)))
    #     )
