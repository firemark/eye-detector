from glob import glob
from shutil import rmtree
from os import mkdir
from os.path import basename, isfile
from contextlib import suppress

from numpy import uint8
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.measure import label, regionprops

from eye_detector.model import FullModel
from eye_detector.const import CLASSES


FACE_SCALES = [4.0]
EYE_SCALES = [2.0]


def load(filepath):
    return imread(filepath)[:, :, 0:3]


def gen_filepath_to_save(klass, filepath):
    name = basename(filepath)
    return f"middata/transformed_label/{klass}/{name}"


def save(filepath_to_save, img):
    img_to_save = img * 0xFF
    imsave(filepath_to_save, img_to_save.astype(uint8))


if __name__ == "__main__":
    with suppress(FileExistsError):
        mkdir("middata/transformed_label")

    model = FullModel(
        face_scales=FACE_SCALES,
        eye_scales=EYE_SCALES,
    )

    tot = 0
    skip = 0
    succ = 0

    with suppress(KeyboardInterrupt):
        for klass in CLASSES:
            with suppress(FileExistsError):
                mkdir(f"middata/transformed_label/{klass}")

            for filepath in glob(f"indata/to_label/{klass}/*.png"):
                filepath_to_save = gen_filepath_to_save(klass, filepath)
                if isfile(filepath_to_save):
                    skip += 1
                    tot += 1
                    continue
                frame = load(filepath)
                img = model.detect(frame)
                if img is not None:
                    print("\033[92m.\033[0m", flush=True, end="")
                    save(filepath_to_save, img)
                    succ +=  1
                else:
                    print("\033[91mX\033[0m", flush=True, end="")
                tot += 1

    print()
    print("---")
    print("total", tot, sep="\t\t")
    print("skipped", skip, sep="\t\t")
    print("with success", succ, sep="\t")

