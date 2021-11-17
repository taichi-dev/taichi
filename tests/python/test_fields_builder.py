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

    # [shape] 1. Test and validation for assigning fields once
    @ti.kernel
    def assign_field_single():
        for i in range(shape_size_1d):
            x[i] = i

    assign_field_single()
    for i in range(shape_size_1d):
        assert x[i] == i

    # [shape] 2. Test and validation for assigning fields twice
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

    # [shape] 3. Test and validation for Re-assigning variable of field
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
    def assign_field_multiple0(a=2, b=5, c=10):
        # test range-for
        for i in range(shape_size_1d):
            x[i] = i * 2
        for i in range(shape_size_1d):
            y[i] = i + 5
        for i in range(shape_size_1d):
            z[i] = i + 10

    assign_field_multiple0(2, 5, 10)
    for i in range(shape_size_1d):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    @ti.kernel
    def assign_field_multiple1(a, b):
        # test struct-for
        for i in y:
            y[i] += a
        for i in z:
            z[i] -= b

    assign_field_multiple1(5, 5)
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
    test_sizes = [1] # [1, 2, 3]
    size_extend_factor = 1 # 10 ** 3
    # note: currently only consider precison that all platform supported,
    # more detailed here: https://docs.taichi.graphics/lang/articles/basic/type#supported-primitive-types
    field_types = [ti.f32, ti.i32]
    field_sizes = [1] # [1, 5, 10]

    def test_for_single_destroy_multi_fields(test_sizes, size_extend_factor, field_types, field_sizes):
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

    test_for_single_destroy_multi_fields(test_sizes, size_extend_factor, field_types, field_sizes)

    def test_for_multi_destroy_multi_fields(test_sizes, size_extend_factor, field_types, field_sizes): #size_1d_0, size_1d_1):
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

    test_for_multi_destroy_multi_fields(size_1d_0, size_1d_1)

    def test_for_raise_twice_destroy(size_1d):
        fb = ti.FieldsBuilder()
        a = ti.field(ti.f32)
        fb.dense(ti.i, size_1d).place(a)
        c = fb.finalize()
        c.destroy()

        with pytest.raises(InvalidOperationError) as e:
            c.destroy()

    test_for_raise_twice_destroy(10)


'''
Test fields with builder exceeds max.
'''
@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder_exceeds_max():
    shape_size = (4, 4)

    def create_fb():
        fb = ti.FieldsBuilder()
        x = ti.field(ti.f32)
        fb.dense(ti.ij, shape_size).place(x)
        fb.finalize()

    # kMaxNumSnodeTreesLlvm=32 in taichi/inc/constants.h
    # TODO(ysh329): kMaxNumSnodeTreesLlvm=512 not 32 in `taichi/inc/constants.h`.
    #               Can this value `kMaxNumSnodeTreesLlvm` load from `.h` file?
    for _ in range(32):
        create_fb()

    with pytest.raises(RuntimeError) as e:
        create_fb()
    assert 'LLVM backend supports up to 32 snode trees' in e.value.args[0]
