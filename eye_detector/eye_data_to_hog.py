from skimage.transform import resize
from glob import glob
from shutil import rmtree
from os import mkdir
from os.path import basename
import pickle

from numpy import concatenate, uint8
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.measure import label, regionprops

from eye_detector.model import load_window
from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.const import CLASSES


SCALES = [1.5, 2.0, 2.5]
SCALES = [1.5]


def detect_and_generate_heatmap(img, window, scale):
    width, height, _ = img.shape
    size = (int(width / scale), int(height / scale))
    heatmap = compute_heatmap((width, height), window(img))
    return resize(heatmap, (width, height))


def get_img_from_bbox(img, region):
    x1, y1, x2, y2 = region.bbox
    eye = img[x1:x2, y1:y2]
    eye = resize(eye, (64, 64))
    return eye


def generate_and_save(klass, filepath, window):
    img = imread(filepath)[:, :, 0:3]
    heatmap = sum(
        detect_and_generate_heatmap(img, window, scale)
        for scale in SCALES
    ) ** 2
    croped = crop_heatmap(heatmap, 0.5)
    regions = regionprops(label(croped))

    if len(regions) != 2:
        return False

    regions.sort(key=lambda o: o.bbox[0])
    left_eye, right_eye = (get_img_from_bbox(img, r) for r in regions)

    name = basename(filepath)
    img_to_save = concatenate([left_eye, right_eye], axis=1) * 0xFF
    imsave(f"middata/transformed_label/{klass}/{name}", img_to_save.astype(uint8))

    return True


if __name__ == "__main__":
    rmtree("middata/transformed_label", ignore_errors=True)
    mkdir("middata/transformed_label")

    window = load_window()

    tot = 0
    succ = 0

    for klass in CLASSES:
        mkdir(f"middata/transformed_label/{klass}")
        for filepath in glob(f"indata/to_label/{klass}/*.png"):
            is_succ = generate_and_save(klass, filepath, window)
            if is_succ:
                print("\033[92m.\033[0m", flush=True, end="")
                succ +=  1
            else:
                print("\033[91mX\033[0m", flush=True, end="")
            tot += 1

    print()
    print("---")
    print("tot", tot)
    print("succ", succ)

