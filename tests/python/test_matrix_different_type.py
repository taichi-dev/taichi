from pytest import approx

import taichi as ti


# TODO: test more matrix operations
@ti.test()
def test_vector():
    type_list = [ti.f32, ti.i32]

    a = ti.Vector.field(len(type_list), dtype=type_list, shape=())
    b = ti.Vector.field(len(type_list), dtype=type_list, shape=())
    c = ti.Vector.field(len(type_list), dtype=type_list, shape=())

    @ti.kernel
    def init():
        a[None] = [1.0, 3]
        b[None] = [2.0, 4]
        c[None] = a[None] + b[None]

    def verify():
        assert isinstance(a[None][0], float)
        assert isinstance(a[None][1], int)
        assert isinstance(b[None][0], float)
        assert isinstance(b[None][1], int)
        assert c[None][0] == 3.0
        assert c[None][1] == 7

    init()
    verify()


# TODO: Support different element types of Matrix on opengl
@ti.test(require=ti.extension.data64, exclude=ti.opengl)
def test_matrix():
    type_list = [[ti.f32, ti.i32], [ti.i64, ti.f32]]
    a = ti.Matrix.field(len(type_list),
                        len(type_list[0]),
                        dtype=type_list,
                        shape=())
    b = ti.Matrix.field(len(type_list),
                        len(type_list[0]),
                        dtype=type_list,
                        shape=())
    c = ti.Matrix.field(len(type_list),
                        len(type_list[0]),
                        dtype=type_list,
                        shape=())

    @ti.kernel
    def init():
        a[None] = [[1.0, 3], [1, 3.0]]
        b[None] = [[2.0, 4], [-2, -3.0]]
        c[None] = a[None] + b[None]

    def verify():
        assert isinstance(a[None][0], float)
        assert isinstance(a[None][1], int)
        assert isinstance(b[None][0], float)
        assert isinstance(b[None][1], int)
        assert c[None][0, 0] == 3.0
        assert c[None][0, 1] == 7
        assert c[None][1, 0] == -1
        assert c[None][1, 1] == 0.0

    init()
    verify()


@ti.test(require=ti.extension.quant_basic)
def test_custom_type():
    cit1 = ti.quant.int(bits=10, signed=True)
    cft1 = ti.type_factory.custom_float(cit1, scale=0.1)
    cit2 = ti.quant.int(bits=22, signed=False)
    cft2 = ti.type_factory.custom_float(cit2, scale=0.1)
    type_list = [[cit1, cft2], [cft1, cit2]]
    a = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    b = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    c = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(a(0, 0), a(0, 1))
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(a(1, 0), a(1, 1))
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(b(0, 0), b(0, 1))
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(b(1, 0), b(1, 1))
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(c(0, 0), c(0, 1))
    ti.root.dense(ti.i, 1).bit_struct(num_bits=32).place(c(1, 0), c(1, 1))

    @ti.kernel
    def init():
        a[0] = [[1, 3.], [2., 1]]
        b[0] = [[2, 4.], [-2., 1]]
        c[0] = a[0] + b[0]

    def verify():
        assert c[0][0, 0] == approx(3, 1e-3)
        assert c[0][0, 1] == approx(7.0, 1e-3)
        assert c[0][1, 0] == approx(0, 1e-3)
        assert c[0][1, 1] == approx(2, 1e-3)

    init()
    verify()
