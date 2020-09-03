import taichi as ti
import numpy as np


@ti.all_archs
def test_fill_scalar():
    val = ti.field(ti.i32)
    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            val[i, j] = i + j * 3

    val.fill(2)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == 2


@ti.all_archs
def test_fill_matrix_scalar():
    val = ti.Matrix.field(2, 3, ti.i32)

    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            for p in range(2):
                for q in range(3):
                    val[i, j][p, q] = i + j * 3

    val.fill(2)

    for i in range(n):
        for j in range(m):
            for p in range(2):
                for q in range(3):
                    assert val[i, j][p, q] == 2


@ti.all_archs
def test_fill_matrix_matrix():
    val = ti.Matrix.field(2, 3, ti.i32)

    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            for p in range(2):
                for q in range(3):
                    val[i, j][p, q] = i + j * 3

    mat = ti.Matrix([[0, 1, 2], [2, 3, 4]])

    val.fill(mat)

    for i in range(n):
        for j in range(m):
            for p in range(2):
                for q in range(3):
                    assert val[i, j][p, q] == mat.get_entry(p, q)


@ti.all_archs
def test_fill_taichi_scope():
    x = ti.field(ti.f32, (5, 7))
    y = ti.Matrix.field(2, 3, ti.i32, (7, 3))

    x.from_numpy(np.random.rand(5, 7))
    y.from_numpy(np.random.randint(10, 100, (7, 3, 2, 3)))

    @ti.kernel
    def func():
        x.fill(2.3)
        for i in range(5):
            x[i, i] -= 0.6
        y.fill(2)

    func()
    assert ti.approx(x.to_numpy()) == np.ones((5, 7), np.int32) * 2.3 - np.eye(5, 7, dtype=np.int32) * 0.6
    assert ti.approx(y.to_numpy()) == np.ones((7, 3, 2, 3), np.float32) * 2
