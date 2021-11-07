import pytest
from taichi.lang.exception import InvalidOperationError

import taichi as ti

@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.metal])
def test_fields_with_shape():
    n = 5
    x = ti.field(ti.f32, [n])

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i

    func()

    for i in range(n):
        assert x[i] == i

    y = ti.field(ti.f32, [n])

    @ti.kernel
    def func2():
        for i in range(n):
            y[i] = i * 2
        for i in range(n):
            x[i] = i * 3

    func2()

    for i in range(n):
        assert x[i] == i * 3
        assert y[i] == i * 2

    func()

    for i in range(n):
        assert x[i] == i


@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.metal])
def test_fields_builder_dense():
    n = 5

    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.dense(ti.i, n).place(x)
    fb1.finalize()

    @ti.kernel
    def func1():
        for i in range(n):
            x[i] = i * 3

    func1()
    for i in range(n):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.dense(ti.i, n).place(y)
    z = ti.field(ti.f32)
    fb2.dense(ti.i, n).place(z)
    fb2.finalize()

    @ti.kernel
    def func2():
        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            y[i] = i + 5
        for i in range(n):
            z[i] = i + 10

    func2()
    for i in range(n):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    func1()
    for i in range(n):
        assert x[i] == i * 3


@ti.test(arch=[ti.cpu, ti.cuda, ti.metal])
def test_fields_builder_pointer():
    n = 5

    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.pointer(ti.i, n).place(x)
    fb1.finalize()

    @ti.kernel
    def func1():
        for i in range(n):
            x[i] = i * 3

    func1()
    for i in range(n):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.pointer(ti.i, n).place(y)
    z = ti.field(ti.f32)
    fb2.pointer(ti.i, n).place(z)
    fb2.finalize()

    # test range-for
    @ti.kernel
    def func2():
        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            y[i] = i + 5
        for i in range(n):
            z[i] = i + 10

    func2()
    for i in range(n):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    # test struct-for
    @ti.kernel
    def func3():
        for i in y:
            y[i] += 5
        for i in z:
            z[i] -= 5

    func3()
    for i in range(n):
        assert y[i] == i + 10
        assert z[i] == i + 5

    func1()
    for i in range(n):
        assert x[i] == i * 3

@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_fields_builder_destroy():

    def test_for_single_destroy_multi_fields(test_sizes, size_extend_factor, field_types, field_sizes):
        fb = ti.FieldsBuilder()
        for tsize_idx in range(len(test_sizes)):
            for ftype_idx in range(len(field_types)):
                for fsize_idx in range(len(field_sizes)):
                    # init
                    test_1d_size = test_sizes[tsize_idx] * size_extend_factor
                    field_type = field_types[ftype_idx]
                    field_size = field_sizes[fsize_idx]

                    for create_field_idx in range(field_size):
                        # TODO: Consider Vector/Matrix, here is Scalar field only.
                        field = ti.field(field_type)
                        fb.dense(ti.i, test_1d_size).place(field)
                    fb_snode_tree = fb.finalize()
                    fb_snode_tree.destroy()

    # [destroy] Args of main body
    test_sizes = [1] # [1, 2, 3]
    size_extend_factor = 1 # 10 ** 3
    # note: currently only consider preicison all paltform supported,
    # more detailed here: https://docs.taichi.graphics/lang/articles/basic/type#supported-primitive-types
    field_types = [ti.f32, ti.i32]
    field_sizes = [1] # [1, 5, 10]

    # [destroy] 1. test for single destroy multiple fields
    test_for_single_destroy_multi_fields(test_sizes, size_extend_factor, field_types, field_sizes)


    def test_for_multi_destroy_multi_fields(size_1d_0, size_1d_1):
        # create 1st field using 1st field builder
        fb0 = ti.FieldsBuilder()
        a0 = ti.field(ti.f64)
        fb0.dense(ti.i, size_1d_0).place(a0)
        c0 = fb0.finalize()

        # create 2nd field using 2nd field builder
        fb1 = ti.FieldsBuilder()
        a1 = ti.field(ti.f64)
        fb1.pointer(ti.i, size_1d_1).place(a1)
        c1 = fb1.finalize()

        # destroy
        c0.destroy()
        c1.destroy()

    def test_for_raise_twice_destroy(size_1d):
        fb = ti.FieldsBuilder()
        a = ti.field(ti.f32)
        fb.dense(ti.i, size_1d).place(a)
        print(11111)
        c = fb.finalize()
        print(22222)
        c.destroy()
        print(33333)
        c.destroy()
        print(444444)
        try:
            c.destroy()
            print(55555)
        except InvalidOperationError:
            print("catched ")
        print(66666)

    # [destroy] 2. test for multiple destroy for multiple fields
#    for size_idx in range(FOR_LOOP_RANGE):
#        size_1d_0 = size_idx * SIZE_EXTEND_FACTOR
#        size_1d_1 = size_idx * SIZE_EXTEND_FACTOR
#        test_for_multi_destroy_multi_fields(size_1d_0, size_1d_1)

    # [destroy] 3. test for raising second destroy
#    for size_idx in range(len(SIZE_CASES)):
#        for field_idx in range(len(SIZE_FIELDS)):
#            # init
#            size_1d = SIZE_CASES[size_idx]
#            size_fields = SIZE_FIELDS[field_idx]

            # start
#            test_for_raise_twice_destroy(size_1d)

@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder_exceeds_max():
    sz = 4

    def create_fb():
        fb = ti.FieldsBuilder()
        x = ti.field(ti.f32)
        fb.dense(ti.ij, (sz, sz)).place(x)
        fb.finalize()

    # kMaxNumSnodeTreesLlvm=32 in taichi/inc/constants.h
    for _ in range(32):
        create_fb()

    with pytest.raises(RuntimeError) as e:
        create_fb()
    assert 'LLVM backend supports up to 32 snode trees' in e.value.args[0]
