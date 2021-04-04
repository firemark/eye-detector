class HogWindow:

    def __init__(self, *, hog, model, patch_size=None):
        self.hog = hog
        self.model = model
        self.patch_size = patch_size or (7, 7)

    def __call__(self, image):
        hoged = self.hog(image)
        img_w, img_h = image.shape[0:2]
        p1w = round(img_w * 0.01)
        p1h = round(img_h * 0.01)
        for (x, y), window in self.sliding_window(hoged):
            to_predict = window.ravel()
            score = self.model.eye_probability(to_predict)
            w, h = self.hog.pixels_per_cell
            a = x * w
            b = y * h
            x1 = max(a - p1w, 0)
            y1 = max(b - p1h, 0)
            x2 = min(a + window.shape[0] * w + p1w, img_w)
            y2 = min(b + window.shape[1] * h + p1h, img_h)
            yield slice(x1, x2), slice(y1, y2), score

    def sliding_window(self, hoged):
        ww, hh = self.patch_size
        width = hoged.shape[0] - ww
        height = hoged.shape[1] - hh
        for x in range(width):
            for y in range(height):
                window = hoged[x:x + ww, y:y + hh]
                yield (x, y), window
