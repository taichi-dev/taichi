from taichi.lang.kernel_arguments import KernelArgument
from taichi.lang.kernel_impl import TaichiCallableTemplateMapper

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_callable_template_mapper():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    mapper = TaichiCallableTemplateMapper(
        (KernelArgument(ti.template(), ti.template()),
         KernelArgument(ti.template(), ti.template()),
         KernelArgument(ti.template(), ti.template())),
        template_slot_locations=(0, 1, 2))
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 1
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 0, 1))[0] == 2
    assert mapper.lookup((0, 1, 0))[0] == 1

    mapper = TaichiCallableTemplateMapper(
        (KernelArgument(ti.i32, ti.i32), KernelArgument(
            ti.i32, ti.i32), KernelArgument(ti.i32, ti.i32)), ())
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 0
    assert mapper.lookup((0, 0, 0))[0] == 0
    assert mapper.lookup((0, 0, 1))[0] == 0
    assert mapper.lookup((0, 1, 0))[0] == 0

    mapper = TaichiCallableTemplateMapper((KernelArgument(
        ti.i32, ti.i32), KernelArgument(
            ti.template(), ti.template()), KernelArgument(ti.i32, ti.i32)),
                                          (1, ))
    assert mapper.lookup((0, x, 0))[0] == 0
    assert mapper.lookup((0, y, 0))[0] == 1
    assert mapper.lookup((0, x, 1))[0] == 0


@test_utils.test()
def test_callable_template_mapper_numpy():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    annotations = (KernelArgument(ti.template(), ti.template()),
                   KernelArgument(ti.template(), ti.template()),
                   KernelArgument(ti.types.ndarray(), ti.types.ndarray()))

    import numpy as np

    mapper = TaichiCallableTemplateMapper(annotations, (0, 1, 2))
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 3),
                                        dtype=np.float32)))[0] == 0
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 4),
                                        dtype=np.float32)))[0] == 0
    assert mapper.lookup((0, 0, np.ones(shape=(1, 2, 1),
                                        dtype=np.int32)))[0] == 1
