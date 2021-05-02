from skimage.transform import resize


class ImgWindow:

    def __init__(self, *, transform, model, patch_size, step):
        self.transform = transform
        self.model = model
        self.patch_size = patch_size
        self.step = step

    def __call__(self, image, scale=1.0):

        if scale != 1.0:
            h, w = image.shape[0:2]
            image = resize(image, (h // scale, w // scale))

        for (x, y), window in self.sliding_window(image):
            to_predict = window.ravel()
            to_predict = to_predict.reshape(1, to_predict.shape[0])
            score = self.model.eye_probability(to_predict)
            a = x * scale
            b = y * scale
            x1 = int(a)
            y1 = int(b)
            x2 = int(a + window.shape[0] * scale)
            y2 = int(b + window.shape[1] * scale)
            yield slice(x1, x2), slice(y1, y2), score

    def get_window_size(self, scale=1.0):
        h, w = self.patch_size
        return w * scale, h * scale

    def sliding_window(self, image):
        step = self.step
        ww, hh = self.patch_size
        width = image.shape[0] - ww
        height = image.shape[1] - hh
        for x in range(0, width, step):
            for y in range(0, height, step):
                window = image[x:x + ww, y:y + hh]
                yield (x, y), self.transform(window)
