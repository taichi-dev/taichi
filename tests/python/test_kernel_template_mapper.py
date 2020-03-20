import taichi as ti


@ti.all_archs
def test_kernel_template_mapper():
    x = ti.var(ti.i32)
    y = ti.var(ti.f32)

    @ti.layout
    def layout():
        ti.root.place(x, y)

    mapper = ti.KernelTemplateMapper(
        (ti.template(), ti.template(), ti.template()),
        template_slot_locations=(0, 1, 2))
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 1
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 0, 1))[0] == 2
    assert mapper.lookup((0, 1, 0))[0] == 1

    mapper = ti.KernelTemplateMapper((ti.i32, ti.i32, ti.i32), ())
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 0
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 0, 1))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 0

    mapper = ti.KernelTemplateMapper((ti.i32, ti.template(), ti.i32), (1, ))
    assert mapper.lookup((0, x, 0))[0] == 0
    assert mapper.lookup((0, y, 0))[0] == 1
    assert mapper.lookup((0, x, 1))[0] == 0


@ti.all_archs
def test_kernel_template_mapper_numpy():
    x = ti.var(ti.i32)
    y = ti.var(ti.f32)

    @ti.layout
    def layout():
        ti.root.place(x, y)

    annotations = (ti.template(), ti.template(), ti.ext_arr())

    import numpy as np

    mapper = ti.KernelTemplateMapper(annotations, (0, 1, 2))
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 3),
                                        dtype=np.float32)))[0] == 0
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 4),
                                        dtype=np.float32)))[0] == 0
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 1),
                                        dtype=np.int32)))[0] == 1
