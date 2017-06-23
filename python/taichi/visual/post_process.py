import numpy as np


class LDRDisplay:
    def __init__(self, exposure=1.0, adaptive_exposure=True, bloom_strength=0.2, bloom_threshold=2, bloom_radius=0.01,
                 gamma=2.2):
        self.bloom_threshold = bloom_threshold
        self.bloom_radius = bloom_radius
        self.bloom_strength = bloom_strength
        self.exposure = exposure
        self.adaptive_exposure = adaptive_exposure
        self.gamma = gamma

    def process(self, img):
        from scipy.ndimage.filters import gaussian_filter
        if self.adaptive_exposure:
            avg = np.mean(img)
            img *= 0.18 / avg * self.exposure
        else:
            img *= self.exposure
        if self.bloom_radius != 0:
            img = img * (1 - self.bloom_strength) + self.bloom_strength * gaussian_filter(img,
                                                                                          sigma=self.bloom_radius * self.bloom_radius * min(
                                                                                              img.shape[:2]))
        img = np.power(img.clip(0, 1), 1 / self.gamma)
        return img


# Taken from http://filmicgames.com/archives/75,
# and https://github.com/mrdoob/three.js/blob/master/examples/js/SkyShader.js
class FilmicToneMapping:
    def __init__(self, exposure=1.0, adaptive_exposure=True, bloom_threshold=2, bloom_radius=0.01, gamma=2.2):
        self.bloom_threshold = bloom_threshold
        self.bloom_radius = bloom_radius
        self.exposure = exposure
        self.adaptive_exposure = adaptive_exposure
        self.gamma = gamma

    def process(self, img):
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        W = 11.2

        from scipy.ndimage.filters import gaussian_filter
        if self.adaptive_exposure:
            avg = np.mean(img)
            img *= 0.18 / avg * self.exposure
        else:
            img *= self.exposure
        if self.bloom_radius != 0:
            img = img * 0.9 + 0.1 * gaussian_filter(img,
                                                    sigma=self.bloom_radius * self.bloom_radius * min(img.shape[:2]))

        def Uncharted2Tonemap(x):
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

        curr = Uncharted2Tonemap(img)
        whiteScale = 1.0 / Uncharted2Tonemap(W)
        img = curr * whiteScale
        return np.power(img.clip(0, 1), 1 / self.gamma)
