class HogWindow:

    def __init__(self, hog, model, patch_size=None):
        self.hog = hog
        self.model = model
        self.patch_size = patch_size or (7, 7)

    def __call__(self, image):
        hoged = self.hog(image)
        for (x, y), window in self.sliding_window(hoged):
            to_predict = window.ravel()
            to_predict = to_predict.reshape(1, to_predict.shape[0])
            score = self.model.predict(to_predict)
            if score[0] != 1:
                continue
            w, h = self.hog.pixels_per_cell
            x1 = x * w
            y1 = y * h
            x2 = x1 + window.shape[0] * w
            y2 = y1 + window.shape[1] * h
            yield slice(x1, x2), slice(y1, y2)

    def sliding_window(self, hoged):
        ww, hh = self.patch_size
        width = hoged.shape[0] - ww
        height = hoged.shape[1] - hh
        for x in range(width):
            for y in range(height):
                window = hoged[x:x + ww, y:y + hh]
                yield (x, y), window
