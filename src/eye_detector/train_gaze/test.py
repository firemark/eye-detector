from time import monotonic
from math import degrees
import torch
from torch.utils.data import DataLoader


def test_data(dataset, net):
    testloader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=20)
    err = 0.0
    angl_err = 0.0
    time = monotonic()


    with torch.no_grad():
        for data in testloader:
            images, gazes = data
            outputs = net(images)
            for o, g in zip(outputs, gazes):
                err += torch.sum(torch.abs(o - g)).item()
                cos_err = torch.dot(o, g) / (torch.norm(o) * torch.norm(g))
                angl_err += torch.arccos(torch.clamp(cos_err, -0.95, 0.95)).item()

    print('           total:', len(dataset))
    print('      mean error:', err / len(dataset))
    print('angle mean error:', degrees(angl_err) / len(dataset), "°")
    print('            time:', monotonic() - time)