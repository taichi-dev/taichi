import tempfile

import numpy as np
import pytest

import taichi as ti


@pytest.mark.parametrize('dtype', [ti.u8, ti.f32])
@ti.test(arch=ti.get_host_arch_list())
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
        with tempfile.NamedTemporaryFile(suffix='.png') as image_path:
            gui.show(image_path.name)
            image = ti.imread(image_path.name)
            delta = (image - i).sum()
            assert delta == 0, "Expected image difference to be 0 but got {} instead.".format(
                delta)
