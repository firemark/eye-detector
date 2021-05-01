import os
from shutil import rmtree
from contextlib import suppress
from glob import glob

from numpy import uint8
from skimage.transform import resize
from skimage import io
from skimage.measure import label, regionprops
from skimage.color import gray2rgb

from eye_detector.model import load_window
from eye_detector.heatmap import compute_heatmap, crop_heatmap



def find_image(path):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    without_suffix = filename.partition('.')[0]
    eye_info_path = os.path.join(dirname, without_suffix + ".eye")
    with open(eye_info_path) as file:
        line = file.readline() # skip first line
        assert line == "#LX	LY	RX	RY\n"
        eye_info = file.readline().split()
    left_cord = eye_info[0:2]
    right_cord = eye_info[2:4]

    img = io.imread(path)
    cords = (left_cord, right_cord)

    return img, without_suffix, cords


def black_img(img, cord):
    x, y = cord
    x = int(x)
    y = int(y)
    x1 = x - 16
    x2 = x + 16
    y1 = y - 16
    y2 = y + 16
    img[y1:y2, x1:x2] = 0.0


def face_detection(window, img):
    heatmap = compute_heatmap(img.shape[0:2], window(img))
    heatmap **= 2
    croped = crop_heatmap(heatmap, limit_ratio=0.1)

    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return None

    return region.bbox


def crop_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[x1:x2, y1:y2]


if __name__ == "__main__":
    paths = glob("indata/bioid/*.pgm", recursive=True)
    with suppress(OSError):
        rmtree(f"indata/face_data/bioid")
    os.mkdir("indata/face_data/bioid")

    face_window = load_window("face")

    for path in paths:
        img, name, cords = find_image(path)

        face_bbox = face_detection(face_window, gray2rgb(img))

        if face_bbox is None:
            print("ERROR")
            continue

        for cord in cords:
            black_img(img, cord)

        height, width = img.shape[0:2]
        img = crop_img(img, face_bbox)
        img = resize(img, (height * 2, width * 2)) * 0xFF

        filename = f"indata/face_data/bioid/{name}.png"
        io.imsave(filename, img.astype(uint8))
