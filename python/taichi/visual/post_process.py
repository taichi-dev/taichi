import numpy as np
from scipy.ndimage.filters import gaussian_filter

class LDRDisplay:
    def __init__(self, bloom_threshold=2, bloom_radius=0.01):
        self.bloom_threshold = bloom_threshold
        self.bloom_radius = bloom_radius

    def process(self, img):
        avg = np.mean(img)
        img *= 0.18 / avg
        if self.bloom_radius != 0:
            img = img * 0.9 + 0.1 * gaussian_filter(img, sigma=self.bloom_radius * self.bloom_radius * min(img.shape[:2]))
        img = np.power(img.clip(0, 1), 1 / 2.2)
        return img