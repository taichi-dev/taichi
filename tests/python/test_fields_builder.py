import pytest
from taichi.lang.exception import InvalidOperationError

import taichi as ti

'''
Test fields with shape.
'''


@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.metal])
def test_fields_with_shape():
    shape_size_1d = 5
    x = ti.field(ti.f32, shape=shape_size_1d)
    y = ti.field(ti.f32, shape=shape_size_1d)

    @ti.kernel
    def assign_field_single():
        for i in range(shape_size_1d):
            x[i] = i

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i

    @ti.kernel
    def assign_field_multiple():
        for i in range(shape_size_1d):
            y[i] = i * 2
        for i in range(shape_size_1d):
            x[i] = i * 3

    assign_field_multiple()
    for i in range(shape_size_1d):
        assert x[i] == i * 3
        assert y[i] == i * 2

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i


'''
Test fields with builder dense.
'''


@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.metal])
def test_fields_builder_dense():
    shape_size_1d = 5
    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.dense(ti.i, shape_size_1d).place(x)
    fb1.finalize()

    @ti.kernel
    def assign_field_single():
        for i in range(shape_size_1d):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.dense(ti.i, shape_size_1d).place(y)
    z = ti.field(ti.f32)
    fb2.dense(ti.i, shape_size_1d).place(z)
    fb2.finalize()

    @ti.kernel
    def assign_field_multiple():
        for i in range(shape_size_1d):
            x[i] = i * 2
        for i in range(shape_size_1d):
            y[i] = i + 5
        for i in range(shape_size_1d):
            z[i] = i + 10

    assign_field_multiple()
    for i in range(shape_size_1d):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i * 3


'''
Test fields with builder pointer.
'''


@ti.test(arch=[ti.cpu, ti.cuda, ti.metal])
def test_fields_builder_pointer():
    shape_size_1d = 5
    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.pointer(ti.i, shape_size_1d).place(x)
    fb1.finalize()

    @ti.kernel
    def assign_field_single():
        for i in range(shape_size_1d):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.pointer(ti.i, shape_size_1d).place(y)
    z = ti.field(ti.f32)
    fb2.pointer(ti.i, shape_size_1d).place(z)
    fb2.finalize()

    @ti.kernel
    def assign_field_multiple0():
        # test range-for
        for i in range(shape_size_1d):
            x[i] = i * 2
        for i in range(shape_size_1d):
            y[i] = i + 5
        for i in range(shape_size_1d):
            z[i] = i + 10

    assign_field_multiple0()
    for i in range(shape_size_1d):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    @ti.kernel
    def assign_field_multiple1():
        # test struct-for
        for i in y:
            y[i] += 5
        for i in z:
            z[i] -= 5

    assign_field_multiple1()
    for i in range(shape_size_1d):
        assert y[i] == i + 10
        assert z[i] == i + 5

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i * 3


'''
Test fields with builder destory.
'''


@ti.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_fields_builder_destroy():
    test_sizes = [1]  # [1, 2, 3]
    size_extend_factor = 1  # 10 ** 3
    # note: currently only consider precison that all platform supported,
    # more detailed here: https://docs.taichi.graphics/lang/articles/basic/type#supported-primitive-types
    field_types = [ti.f32, ti.i32]
    field_sizes = [1]  # [1, 5, 10]

    def test_for_single_destroy_multi_fields(test_sizes, size_extend_factor,
                                             field_types, field_sizes):
        fb = ti.FieldsBuilder()
        for tsize_idx in range(len(test_sizes)):
            for ftype_idx in range(len(field_types)):
                for fsize_idx in range(len(field_sizes)):
                    test_1d_size = test_sizes[tsize_idx] * size_extend_factor
                    field_type = field_types[ftype_idx]
                    field_size = field_sizes[fsize_idx]

                    for create_field_idx in range(field_size):
                        field = ti.field(field_type)
                        fb.dense(ti.i, test_1d_size).place(field)
        fb_snode_tree = fb.finalize()
        fb_snode_tree.destroy()

    test_for_single_destroy_multi_fields(test_sizes, size_extend_factor,
                                         field_types, field_sizes)

    def test_for_multi_destroy_multi_fields(test_sizes, size_extend_factor,
                                            field_types, field_sizes):
        fb0 = ti.FieldsBuilder()
        fb1 = ti.FieldsBuilder()

        for tsize_idx in range(len(test_sizes)):
            for ftype_idx in range(len(field_types)):
                for fsize_idx in range(len(field_sizes)):
                    test_1d_size = test_sizes[tsize_idx] * size_extend_factor
                    field_type = field_types[ftype_idx]
                    field_size = field_sizes[fsize_idx]

                    for create_field_idx in range(field_size):
                        field0 = ti.field(field_type)
                        field1 = ti.field(field_type)

                        fb0.dense(ti.i, test_1d_size).place(field0)
                        fb1.pointer(ti.i, test_1d_size).place(field1)

                        fb0_snode_tree = fb0.finalize()
                        fb1_snode_tree = fb1.finalize()

        # destroy
        fb0_snode_tree.destroy()
        fb1_snode_tree.destroy()

    test_for_multi_destroy_multi_fields(test_sizes, size_extend_factor,
                                        field_types, field_sizes)

    def test_for_raise_twice_destroy(size_1d):
        fb = ti.FieldsBuilder()
        a = ti.field(ti.f32)
        fb.dense(ti.i, size_1d).place(a)
        c = fb.finalize()

        with pytest.raises(InvalidOperationError) as e:
            # Triggered if destroy twice
            c.destroy()

    test_for_raise_twice_destroy(10)
