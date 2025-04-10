from glob import glob
from os import mkdir
from contextlib import suppress
from posixpath import basename, isfile

from skimage.io import imread, imsave

from eye_detector.const import CLASSES
from eye_detector.dlib_model import Model


def load(filepath):
    return imread(filepath)[:, :, 0:3]


def gen_filepath_to_save(klass, filepath):
    name = basename(filepath)
    return f"middata/transformed_label/{klass}/{name}"


def save(filepath_to_save, img):
    img_to_save = img * 0xFF
    imsave(filepath_to_save, img_to_save.astype(np.uint8))


if __name__ == "__main__":
    with suppress(FileExistsError):
        mkdir("middata/transformed_label")

    model = Model()
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

