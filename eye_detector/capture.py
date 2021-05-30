import pickle
from sys import argv
from time import time

import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from eye_detector.model import FullModel
from eye_detector.cam_func import init_win, del_win, draw_it
from eye_detector.train_net import Net, transform
from eye_detector.const import CLASSES


def draw_region(frame, region, color):
    if region is None:
        return
    y1, x1, y2, x2 = region.bbox
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )


def get_net_label(eyes_img, net):
    e_img = ToPILImage()((eyes_img * 255).astype(np.uint8))
    t_img = transform(e_img)
    t_img = t_img.reshape([1, *t_img.shape])
    outputs = net(t_img)
    _, it = torch.max(outputs.data, 1)
    return it


def load_net():
    net = Net()
    net_state = torch.load('outdata/net.pth')
    net.load_state_dict(net_state)
    return net


def main():
    net = load_net()
    full_model = FullModel(
        face_scales=[4.0],
        eye_scales=[2.0],
    )
    cap = init_win()
    i = 0

    while True:
        ret, frame = cap.read()

        t = time()

        faces_region, eyes_region, eyes_img = full_model.detect(
            frame=frame,
            with_debug_regions=True,
        )
        draw_region(frame, faces_region, color=(0xFF, 0x50, 0x50))
        draw_region(frame, eyes_region, color=(0x00, 0x50, 0xFF))
        if eyes_img is not None:
            it = get_net_label(eyes_img, net)
            draw_it(frame, it)
            print(it, CLASSES[it])

        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)


if __name__ == "__main__":
    main()
