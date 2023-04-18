"""Physarum simulation example.

See https://sagejenson.com/physarum for the details."""

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

PARTICLE_N = 1024
GRID_SIZE = 512
SENSE_ANGLE = 0.20 * np.pi
SENSE_DIST = 4.0
EVAPORATION = 0.95
MOVE_ANGLE = 0.1 * np.pi
MOVE_STEP = 2.0

grid = ti.field(dtype=ti.f32, shape=[2, GRID_SIZE, GRID_SIZE])
position = ti.Vector.field(2, dtype=ti.f32, shape=[PARTICLE_N])
heading = ti.field(dtype=ti.f32, shape=[PARTICLE_N])


@ti.kernel
def init():
    for p in ti.grouped(grid):
        grid[p] = 0.0
    for i in position:
        position[i] = ti.Vector([ti.random(), ti.random()]) * GRID_SIZE
        heading[i] = ti.random() * np.pi * 2.0


@ti.func
def sense(phase, pos, ang):
    p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * SENSE_DIST
    return grid[phase, p.cast(int) % GRID_SIZE]


@ti.kernel
def step(phase: ti.i32):
    # move
    for i in position:
        pos, ang = position[i], heading[i]
        l = sense(phase, pos, ang - SENSE_ANGLE)
        c = sense(phase, pos, ang)
        r = sense(phase, pos, ang + SENSE_ANGLE)
        if l < c < r:
            ang += MOVE_ANGLE
        elif l > c > r:
            ang -= MOVE_ANGLE
        elif c < l and c < r:
            ang += MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
        pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * MOVE_STEP
        position[i], heading[i] = pos, ang

    # deposit
    for i in position:
        ipos = position[i].cast(int) % GRID_SIZE
        grid[phase, ipos] += 1.0

    # diffuse
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        a = 0.0
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                a += grid[phase, (i + di) % GRID_SIZE, (j + dj) % GRID_SIZE]
        a *= EVAPORATION / 9.0
        grid[1 - phase, i, j] = a


def main():
    print("[Hint] Use slider to change simulation speed.")
    gui = ti.GUI("Physarum")
    init()
    i = 0
    step_per_frame = gui.slider("step_per_frame", 1, 100, 1)
    while gui.running and not gui.get_event(gui.ESCAPE):
        for _ in range(int(step_per_frame.value)):
            step(i % 2)
            i += 1
        gui.set_image(grid.to_numpy()[0])
        gui.show()


if __name__ == "__main__":
    main()
