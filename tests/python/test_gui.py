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


@pytest.mark.parametrize("vector_field", [True, False])
@pytest.mark.parametrize("dtype", [ti.u8, ti.f32, ti.f64])
@pytest.mark.parametrize("color", [0, 32, 64, 128, 255])
@pytest.mark.parametrize("offset", [None, (-150, -150), (0, 0), (150, 150)])
@test_utils.test(arch=get_host_arch_list())
def test_set_image_with_offset(vector_field, dtype, color, offset):
    n = 300
    shape = (n, n)

    img = (
        ti.Vector.field(dtype=dtype, n=3, shape=shape, offset=offset)
        if vector_field
        else ti.field(dtype=dtype, shape=shape, offset=offset)
    )
    img.fill(color if dtype is ti.u8 else color * 1.0 / 255)

    gui = ti.GUI(name="test", res=shape, show_gui=False, fast_gui=False)
    gui.set_image(img)

    image_path = test_utils.make_temp_file(suffix=".png")
    gui.show(image_path)
    image = ti.tools.imread(image_path)
    delta = (image - color).sum()
    assert delta == 0, "Expected image difference to be 0 but got {} instead.".format(delta)


@pytest.mark.parametrize("channel", [3, 4])
@pytest.mark.parametrize("dtype", [ti.u8, ti.f32, ti.f64])
@pytest.mark.parametrize("color", [0, 32, 64, 128, 255])
@pytest.mark.parametrize("offset", [None, (-150, -150), (0, 0), (150, 150)])
@test_utils.test(arch=get_host_arch_list())
def test_set_image_fast_gui_with_offset(channel, dtype, color, offset):
    n = 300
    shape = (n, n)

    img = ti.Vector.field(dtype=dtype, n=channel, shape=shape, offset=offset)
    img.fill(color if dtype is ti.u8 else color * 1.0 / 255)

    gui = ti.GUI(name="test", res=shape, show_gui=False, fast_gui=True)
    gui.set_image(img)
    fast_image = gui.img

    alpha = 0xFF << 24
    from taichi._lib.utils import get_os_name  # pylint: disable=C0415

    rgb_color = (
        (color << 16) + (color << 8) + color
        if ti.static(get_os_name() != "osx")
        else (color << 16) + (color << 8) + color + alpha
    )
    ground_truth = np.full(n * n, rgb_color, dtype=np.uint32)

    assert np.allclose(fast_image, ground_truth)
