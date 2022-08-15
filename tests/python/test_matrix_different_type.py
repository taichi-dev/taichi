from pytest import approx

import taichi as ti
from tests import test_utils


# TODO: test more matrix operations
@test_utils.test()
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
@test_utils.test(require=ti.extension.data64, exclude=ti.opengl)
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
        assert isinstance(a[None][0, 0], float)
        assert isinstance(a[None][0, 1], int)
        assert isinstance(b[None][0, 0], float)
        assert isinstance(b[None][0, 1], int)
        assert c[None][0, 0] == 3.0
        assert c[None][0, 1] == 7
        assert c[None][1, 0] == -1
        assert c[None][1, 1] == 0.0

    init()
    verify()


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_type():
    qit1 = ti.types.quant.int(bits=10, signed=True)
    qfxt1 = ti.types.quant.fixed(bits=10, signed=True, scale=0.1)
    qit2 = ti.types.quant.int(bits=22, signed=False)
    qfxt2 = ti.types.quant.fixed(bits=22, signed=False, scale=0.1)
    type_list = [[qit1, qfxt2], [qfxt1, qit2]]
    a = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    b = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    c = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(0, 0), a.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(1, 0), a.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(0, 0), b.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(1, 0), b.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(0, 0), c.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(1, 0), c.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)

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
