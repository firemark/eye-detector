import os
from shutil import rmtree
from contextlib import suppress
from argparse import ArgumentParser
from glob import glob

from numpy import uint8
from skimage.transform import resize
from skimage import io
from skimage.measure import label, regionprops
from skimage.color import gray2rgb

from eye_detector.model import load_window
from eye_detector.heatmap import compute_heatmap, crop_heatmap

parser = ArgumentParser(
    description="transform face to eyenose/eye train",
)
parser.add_argument(
    "type",
    metavar="TYPE",
    choices=['FACE', 'EYENOSE'],
)


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


def detection(window, img, limit_ratio):
    heatmap = compute_heatmap(img.shape[0:2], window(img))
    heatmap **= 2
    croped = crop_heatmap(heatmap, limit_ratio)

    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return None

    return region.bbox


def crop_img(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[x1:x2, y1:y2]


def face_detection(windows, img):
    return detection(windows['face'], img, limit_ratio=0.3)


def eyenose_detection(windows, img):
    bbox = face_detection(windows, img)
    if bbox is None:
        return None
    img = crop_img(img, bbox)
    return detection(windows['eyenose'], img, limit_ratio=0.5)

detects = {
    'face': face_detection,
    'eyenose': eyenose_detection,
}


if __name__ == "__main__":
    args = parser.parse_args()
    type_name = args.type.lower()

    paths = glob("indata/bioid/*.pgm", recursive=True)
    with suppress(OSError):
        rmtree(f"indata/face_data/bioid_{type_name}")
    os.mkdir(f"indata/face_data/bioid_{type_name}")

    windows = {'face': load_window('face')}
    if type_name == 'eyenose':
        windows['eyenose'] = load_window('eyenose')

    detect = detects[type_name]

    for path in paths:
        img, name, cords = find_image(path)

        bbox = detect(windows, gray2rgb(img))

        if bbox is None:
            print("ERROR")
            continue

        for cord in cords:
            black_img(img, cord)

        img = crop_img(img, bbox)
        height, width = img.shape[0:2]
        img = resize(img, (height * 2, width * 2)) * 0xFF

        filename = f"indata/face_data/bioid_{type_name}/{name}.png"
        io.imsave(filename, img.astype(uint8))
