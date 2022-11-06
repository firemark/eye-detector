import numpy as np
import torch
from torch.utils.data import DataLoader


def test_data(dataset, net):
    testloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    err = 0.0

    with torch.no_grad():
        for data in testloader:
            images, gazes = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for g, p in zip(gazes, predicted):
                err += sum(np.abs(g - p)).item()

    print('---')
    print('     total:', len(dataset))
    print('mean error:', err / len(dataset))
    # print(' '.join('%12s' % label for label in ('-',) + CLASSES))
    # for y, label in enumerate(CLASSES):
    #     print(
    #         '%12s' % label,
    #         ' '.join('%12d' % matrix[x, y] for x in range(len(CLASSES)))
    #     )
