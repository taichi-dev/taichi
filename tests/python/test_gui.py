import numpy as np
import pytest
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.u8, ti.f32])
@test_utils.test(arch=get_host_arch_list())
def test_save_image_without_window(dtype):
    n = 255
    pixels = ti.field(dtype=dtype, shape=(n, n, 3))

    @ti.kernel
    def paint(c: dtype):
        for i, j, k in pixels:
            pixels[i, j, k] = c

    gui = ti.GUI("Test", res=(n, n), show_gui=False)
    for i in [0, 32, 64, 128, 255]:
        if dtype is ti.u8:
            paint(i)
        else:
            paint(i * 1.0 / n)
        gui.set_image(pixels)
        image_path = test_utils.make_temp_file(suffix=".png")
        gui.show(image_path)
        image = ti.tools.imread(image_path)
        delta = (image - i).sum()
        assert delta == 0, "Expected image difference to be 0 but got {} instead.".format(delta)


@pytest.mark.parametrize("fast_gui", [True, False])
@pytest.mark.parametrize("dtype, color", [(ti.u8, 128), (ti.f32, 0.5), (ti.f64, 0.5)])
@pytest.mark.parametrize("offset", [(-299, -299), (-150, -150), (0, 0), (150, 150), (299, 299)])
@test_utils.test(arch=get_host_arch_list())
def test_set_image_with_offset(fast_gui, offset, dtype, color):
    n = 300
    shape = (n, n)
    if fast_gui is True or dtype is ti.f64:
        img = ti.Vector.field(dtype=dtype, n=3, shape=shape, offset=offset)
    else:
        img = ti.field(dtype=dtype, shape=shape, offset=offset)
    img.fill(color)

    gui = ti.GUI(name="test", res=shape, show_gui=False, fast_gui=fast_gui)
    gui.set_image(img)
    gui.show()
