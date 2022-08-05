import argparse
import os

from taichi.examples.patterns import taichi_logo

import taichi as ti

ti.init(arch=ti.cuda, debug=True)

n = 512
x = ti.field(dtype=ti.i32)
res = n + n // 4 + n // 16 + n // 64
img = ti.field(dtype=ti.f32, shape=(res, res))

block1 = ti.root.pointer(ti.ij, n // 64)
block2 = block1.pointer(ti.ij, 4)
block3 = block2.pointer(ti.ij, 4)
block3.dense(ti.ij, 4).place(x)


@ti.kernel
def check_img_value():
    s: ti.f32 = 0
    for i in ti.static(range(20)):
        s += img[i, i]
    assert s == 15.25


@ti.kernel
def fill_img():
    img.fill(0.05)


@ti.kernel
def block1_deactivate_all():
    for I in ti.grouped(block3):
        ti.deactivate(block3, I)

    for I in ti.grouped(block2):
        ti.deactivate(block2, I)

    for I in ti.grouped(block1):
        ti.deactivate(block1, I)


@ti.kernel
def activate(t: ti.f32):
    for i, j in ti.ndrange(n, n):
        p = ti.Vector([i, j]) / n
        p = ti.Matrix.rotation2d(ti.sin(t)) @ (p - 0.5) + 0.5

        if taichi_logo(p) == 0:
            x[i, j] = 1


@ti.func
def scatter(i):
    return i + i // 4 + i // 16 + i // 64 + 2


@ti.kernel
def paint():
    for i, j in ti.ndrange(n, n):
        t = x[i, j]
        block1_index = ti.rescale_index(x, block1, [i, j])
        block2_index = ti.rescale_index(x, block2, [i, j])
        block3_index = ti.rescale_index(x, block3, [i, j])
        t += ti.is_active(block1, block1_index)
        t += ti.is_active(block2, block2_index)
        t += ti.is_active(block3, block3_index)
        img[scatter(i), scatter(j)] = 1 - t / 4


def save_kernels(arch):
    m = ti.aot.Module(arch)

    m.add_kernel(fill_img, template_args={})
    m.add_kernel(block1_deactivate_all, template_args={})
    m.add_kernel(activate, template_args={})
    m.add_kernel(paint, template_args={})
    m.add_kernel(check_img_value, template_args={})

    m.add_field("x", x)
    m.add_field("img", img)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m.save(dir_name, 'whatever')


if __name__ == '__main__':
    save_kernels(arch=ti.cuda)
