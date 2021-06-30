import math

import taichi as ti

@ti.func
def randUnit2D():
    '''
    Generate a 2-D random unit vector whose length is equal to 1.0.
    The return value is a 2-D vector, whose tip distributed evenly
    **on the border** of a unit circle.
    :return:
        The return value is computed as::
            a = rand() * math.tau
            return vec(ti.cos(a), ti.sin(a))
    '''
    a = ti.random() * math.tau
    return ti.Vector([ti.cos(a), ti.sin(a)])


@ti.func
def randUnit3D():
    '''
    Generate a 3-D random unit vector whose length is equal to 1.0.
    The return value is a 3-D vector, whose tip distributed evenly
    **on the surface** of a unit sphere.
    :return:
        The return value is computed as::
            u = randUnit2D()
            s = rand() * 2 - 1
            c = sqrt(1 - s ** 2)
            return vec3(c * u, s)
    '''
    u = randUnit2D()
    s = ti.random() * 2 - 1
    c = ti.sqrt(1 - s ** 2)
    return ti.Vector([c * u[0], c * u[1], s])