# Marching squares algorithm
# https://en.wikipedia.org/wiki/Marching_squares

import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

N = 128

edge_table_np = np.array(
    [
        [[-1, -1], [-1, -1]],  # Case 0
        [[3, 0], [-1, -1]],  # Case 1
        [[0, 1], [-1, -1]],  # Case 2
        [[1, 3], [-1, -1]],  # Case 3
        [[1, 2], [-1, -1]],  # Case 4
        [[0, 1], [2, 3]],  # Case 5
        [[0, 2], [-1, -1]],  # Case 6
        [[2, 3], [-1, -1]],  # Case 7
        [[2, 3], [-1, -1]],  # Case 8
        [[0, 2], [-1, -1]],  # Case 9
        [[0, 3], [1, 2]],  # Case 10
        [[1, 2], [-1, -1]],  # Case 11
        [[1, 3], [-1, -1]],  # Case 12
        [[0, 1], [-1, -1]],  # Case 13
        [[3, 0], [-1, -1]],  # Case 14
        [[-1, -1], [-1, -1]],  # Case 15
    ],
    dtype=np.int32)

pixels = ti.field(float, (N, N))

edge_coords = ti.Vector.field(2, float, (N**2, 2))  # generated edges
edge_table = ti.Vector.field(2, int, (16, 2))  # edge table (constant)
edge_table.from_numpy(edge_table_np)


@ti.func
def gauss(x, sigma):
    # Un-normalized Gaussian distribution
    return ti.exp(-x**2 / (2 * sigma**2))


@ti.kernel
def touch(mx: float, my: float, size: float):
    for I in ti.grouped(pixels):
        mouse_pos = ti.Vector([mx, my]) + 0.5 / N
        peak_center = gauss((I / N - 0.5).norm(), size)
        peak_mouse = gauss((I / N - mouse_pos).norm(), size)
        pixels[I] = peak_center + peak_mouse


@ti.func
def get_vertex(vertex_id):
    # bottom edge
    v = ti.Vector([0.5, 0.0])
    # right edge
    if vertex_id == 1:
        v = ti.Vector([1.0, 0.5])
    # top edge
    elif vertex_id == 2:
        v = ti.Vector([0.5, 1.0])
    # left edge
    elif vertex_id == 3:
        v = ti.Vector([0.0, 0.5])
    return v


@ti.kernel
def march(level: float) -> int:
    n_edges = 0

    for i, j in ti.ndrange(N - 1, N - 1):
        case_id = 0
        if pixels[i, j] > level: case_id |= 1
        if pixels[i + 1, j] > level: case_id |= 2
        if pixels[i + 1, j + 1] > level: case_id |= 4
        if pixels[i, j + 1] > level: case_id |= 8

        for k in range(2):
            if edge_table[case_id, k][0] == -1:
                break

            n = ti.atomic_add(n_edges, 1)
            for l in ti.static(range(2)):
                vertex_id = edge_table[case_id, k][l]
                edge_coords[n, l] = ti.Vector([i, j
                                               ]) + get_vertex(vertex_id) + 0.5

    return n_edges


level = 0.2

gui = ti.GUI('Marching squares')
while gui.running and not gui.get_event(gui.ESCAPE):
    touch(*gui.get_cursor_pos(), 0.05)
    n_edges = march(level)
    edge_coords_np = edge_coords.to_numpy()[:n_edges] / N
    gui.set_image(ti.imresize(pixels, *gui.res) / level)
    gui.lines(edge_coords_np[:, 0],
              edge_coords_np[:, 1],
              color=0xff66cc,
              radius=1.5)
    gui.show()
