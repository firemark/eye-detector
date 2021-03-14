import cv2
import torch
from PIL import Image

from const import CLASSES, ROWS
from const import W, H, PW, PH
from cam_func import init_win, del_win, draw_camera
from train import Net, transform

cap = init_win()
net = Net()
net_state = torch.load('net.pth')
net.load_state_dict(net_state)
it = 0
border_color = 0

while True:
    ret, frame = cap.read()
    img = Image.fromarray(frame)
    try:
        transformed_img = transform(img)
    except:
        border_color = (0x10, 0x20, 0xFF)
    else:
        transformed_img = transformed_img.reshape([1, *transformed_img.shape])
        outputs = net(transformed_img)
        _, it = torch.max(outputs.data, 1)
        border_color = (0, 0xFF, 0)

    print(it)
    draw_camera(frame, it, border_color=border_color)
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'):
        break

del_win(cap)
