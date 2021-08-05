import math

import taichi as ti


@ti.func
def rand_vector(n: ti.template()):
    '''
    Samples a n-dimensional random uniform vector.
    '''
    return ti.Vector([ti.random() for _ in range(n)])


@ti.func
def clamp(x, minval, maxval):
    return min(max(x, minval), maxval)


@ti.func
def rand_unit_2d():
    '''
    Uniformly samples a vector on a 2D unit circle.
    '''
    a = ti.random() * math.tau
    return ti.Vector([ti.cos(a), ti.sin(a)])


@ti.func
def rand_unit_3d():
    '''
    Uniformly samples a vector on a 3D unit sphere.
    '''
    u = rand_unit_2d()
    s = ti.random() * 2 - 1
    c = ti.sqrt(1 - s**2)
    return ti.Vector([c * u[0], c * u[1], s])


@ti.func
def rand_disk_2d():
    '''
    Uniformly samples a point within the area of a 2D unit disk.
    '''
    x = 2 * ti.random() - 1
    y = 2 * ti.random() - 1
    while x * x + y * y > 1:
        x = 2 * ti.random() - 1
        y = 2 * ti.random() - 1
    return ti.Vector([x, y])


@ti.func
def reflect_boundary(pos,
                     vel,
                     pmin=0,
                     pmax=1,
                     rebound=1,
                     rebound_perpendicular=1):
    '''
    Reflects particle velocity from a rectangular boundary (if collides).
    `boundaryReflect` takes particle position, velocity and other parameters.
    '''
    cond = pos < pmin and vel < 0 or pos > pmax and vel > 0
    for j in ti.static(range(pos.n)):
        if cond[j]:
            vel[j] *= -rebound
            for k in ti.static(range(pos.n)):
                if k != j:
                    vel[k] *= rebound_perpendicular
    return vel
