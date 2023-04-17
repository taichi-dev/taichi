"""
Given an input image, redraw it with circle packings.
"""
try:
    import cairo
    import cv2
except:
    raise ImportError(
        "This example depends on opencv and cairo, please run 'pip install opencv-python pycairo' to install."
    )

import os

import matplotlib.pyplot as plt
import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

_internal_scale = 5  # internally we need a large image to paint


@ti.dataclass
class Circle:
    x: int
    y: int
    r: int


circles = Circle.field()  # pylint: disable=no-member
ti.root.dynamic(ti.i, 100000, chunk_size=64).place(circles)


def load_image(imgfile):
    image = cv2.imread(imgfile)
    h, w = image.shape[:2]
    image = cv2.resize(
        image,
        (int(_internal_scale * w), int(_internal_scale * h)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_dist_transform_image(image):
    canny = cv2.Canny(image, 100, 200)
    edges_inv = 255 - canny
    dist_image = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 0)
    return dist_image


@ti.kernel
def add_new_circles(
    filled: ti.types.ndarray(),
    dist_image: ti.types.ndarray(),
    min_radius: int,
    max_radius: int,
) -> int:
    H, W = dist_image.shape[0], dist_image.shape[1]
    ti.loop_config(serialize=True)
    for x in range(min_radius, W - min_radius):
        for y in range(min_radius, H - min_radius):
            valid = True
            if dist_image[y, x] > min_radius:
                r = int((dist_image[y, x] + 1) / 2)
                r = ti.min(r, max_radius)
                if not filled[y, x] and r <= x < W - r and r <= y < H - r:
                    for ii in range(x - r, x + r + 1):
                        for jj in range(y - r, y + r + 1):
                            if (ii - x) ** 2 + (jj - y) ** 2 < (r + 1) ** 2:
                                if filled[jj, ii]:
                                    valid = False
                                    break
                        if not valid:
                            break

                    if valid:
                        circles.append(Circle(x, y, r))
                        for ii in range(x - r, x + r + 1):
                            for jj in range(y - r, y + r + 1):
                                if (ii - x) ** 2 + (jj - y) ** 2 < (r + 1) ** 2:
                                    filled[jj, ii] = 1

    return circles.length()


def plot_cirlces(image, ctx, n):
    for i in range(n):
        c = circles[i]
        fc = image[c.y, c.x] / 255
        if all(fc < 0.1):
            ec = (0.5, 0.5, 0.5)
        else:
            ec = (0, 0, 0)
        ctx.arc(c.x, c.y, c.r, 0, 2 * np.pi)
        ctx.set_source_rgb(*fc)
        ctx.fill_preserve()
        ctx.set_source_rgba(*ec)
        ctx.stroke()


def process(imgfile, scale):
    image = load_image(imgfile)
    dist_image = get_dist_transform_image(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    H, W = image.shape[:2]
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    filled = np.zeros([H, W], dtype=np.int32)
    R = [150, 120, 100, 80, 50, 30, 25, 20, 15, 10, 7, 5, 3, 2]
    for i in range(1, len(R)):
        n = add_new_circles(filled, dist_image, R[i], R[i - 1])

    ctx.set_line_width(1)
    plot_cirlces(image, ctx, n)
    data = surface.get_data()
    result = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4)
    w = int(W / _internal_scale * scale)
    h = int(H / _internal_scale * scale)
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
    plt.tight_layout()
    plt.axis("off")
    plt.imshow(result)
    plt.show()


def main(imgfile=None, scale=2):
    if imgfile is None:
        imgfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "taichi_logo.png")
    process(imgfile, scale)


if __name__ == "__main__":
    main()
