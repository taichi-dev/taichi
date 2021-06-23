from random import randrange

import taichi as ti


@ti.all_archs
def test_listgen():
    x = ti.field(ti.i32)
    n = 1024

    ti.root.dense(ti.ij, 4).dense(ti.ij, 4).dense(ti.ij,
                                                  4).dense(ti.ij,
                                                           4).dense(ti.ij,
                                                                    4).place(x)

    @ti.kernel
    def fill(c: ti.i32):
        for i, j in x:
            x[i, j] = i * 10 + j + c

    for c in range(2):
        print('Testing c=%d' % c)
        fill(c)
        # read it out once to avoid launching too many operator[] kernels
        xnp = x.to_numpy()
        for i in range(n):
            for j in range(n):
                assert xnp[i, j] == i * 10 + j + c

        # Randomly check 1000 items to ensure [] work as well
        for _ in range(1000):
            i, j = randrange(n), randrange(n)
            assert x[i, j] == i * 10 + j + c


@ti.all_archs
def test_nested_3d():
    x = ti.field(ti.i32)
    n = 128

    ti.root.dense(ti.ijk, 4).dense(ti.ijk, 4).dense(ti.ijk,
                                                    4).dense(ti.ijk,
                                                             2).place(x)

    @ti.kernel
    def fill():
        for i, j, k in x:
            x[i, j, k] = (i * n + j) * n + k

    fill()
    # read it out once to avoid launching too many operator[] kernels
    xnp = x.to_numpy()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert xnp[i, j, k] == (i * n + j) * n + k

    # Randomly check 1000 items to ensure [] work as well
    for _ in range(1000):
        i, j, k = randrange(n), randrange(n), randrange(n)
        assert x[i, j, k] == (i * n + j) * n + k
