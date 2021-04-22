from skimage.transform import resize


class HogWindow:

    def __init__(self, *, hog, model, patch_size=None):
        self.hog = hog
        self.model = model
        self.patch_size = patch_size or (7, 7)

    def __call__(self, image, scale=1.0):
        img_w, img_h = image.shape[0:2]

        if scale != 1.0:
            w, h = image.shape[0:2]
            image = resize(image, (w // scale, h // scale))

        hoged = self.hog(image)
        w, h = self.hog.pixels_per_cell
        w *= scale
        h *= scale
        for (x, y), window in self.sliding_window(hoged):
            to_predict = window.ravel()
            score = self.model.eye_probability(to_predict)
            a = x * w
            b = y * h
            x1 = int(max(a, 0))
            y1 = int(max(b, 0))
            x2 = int(min(a + window.shape[0] * w, img_w))
            y2 = int(min(b + window.shape[1] * h, img_h))
            yield slice(x1, x2), slice(y1, y2), score

    def get_window_size(self, scale=1.0):
        w, h = self.hog.pixels_per_cell
        pw, ph = self.patch_size
        return pw * w * scale, ph * h * scale

    def sliding_window(self, hoged):
        ww, hh = self.patch_size
        width = hoged.shape[0] - ww
        height = hoged.shape[1] - hh
        for x in range(width):
            for y in range(height):
                window = hoged[x:x + ww, y:y + hh]
                yield (x, y), window
