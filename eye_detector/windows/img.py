class ImgWindow:

    def __init__(self, *, transform, model, patch_size, step):
        self.transform = transform
        self.model = model
        self.patch_size = patch_size
        self.step = step

    def __call__(self, image):
        for (x, y), window in self.sliding_window(image):
            to_predict = window.ravel()
            to_predict = to_predict.reshape(1, to_predict.shape[0])
            score = self.model.eye_probability(to_predict)
            x1 = x
            y1 = y
            x2 = x + window.shape[0]
            y2 = y + window.shape[1]
            yield slice(x1, x2), slice(y1, y2), score

    def sliding_window(self, image):
        step = self.step
        ww, hh = self.patch_size
        width = image.shape[0] - ww
        height = image.shape[1] - hh
        for x in range(0, width, step):
            for y in range(0, height, step):
                window = image[x:x + ww, y:y + hh]
                yield (x, y), self.transform(window)
