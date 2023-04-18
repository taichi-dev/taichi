import time

import numpy as np

import taichi as ti


def init():
    a = []
    for i in np.linspace(0, 1, n, False):
        for j in np.linspace(0, 1, n, False):
            a.append([i, j])
    return np.array(a).astype(np.float32)


ti.init(arch=ti.gpu)
n = 50
dirs = ti.field(dtype=float, shape=(n * n, 2))
locations_np = init()

locations = ti.field(dtype=float, shape=(n * n, 2))
locations.from_numpy(locations_np)


@ti.kernel
def paint(t: float):
    (o, p) = locations_np.shape
    for i in range(0, o):  # Parallelized over all pixels
        x = locations[i, 0]
        y = locations[i, 1]
        dirs[i, 0] = ti.sin((t * x - y))
        dirs[i, 1] = ti.cos(t * y - x)
        l = (dirs[i, 0] ** 2 + dirs[i, 1] ** 2) ** 0.5
        dirs[i, 0] /= l * 40
        dirs[i, 1] /= l * 40


def main():
    gui = ti.GUI("Vector Field", res=(500, 500))

    beginning = time.time_ns()
    for k in range(1000000):
        paint((time.time_ns() - beginning) * 0.00000001)
        dirs_np = dirs.to_numpy()
        gui.arrows(locations_np, dirs_np, radius=1)
        gui.show()


if __name__ == "__main__":
    main()
