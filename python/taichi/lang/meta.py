from taichi.core import settings
from taichi.lang import impl
from taichi.lang.expr import Expr
from taichi.lang.field import ScalarField
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel

import taichi as ti

# A set of helper (meta)functions


@kernel
def fill_tensor(tensor: template(), val: template()):
    for I in ti.grouped(tensor):
        tensor[I] = val


@kernel
def tensor_to_ext_arr(tensor: template(), arr: ext_arr()):
    for I in ti.grouped(tensor):
        arr[I] = tensor[I]


@kernel
def vector_to_fast_image(img: template(), out: ext_arr()):
    # FIXME: Why is ``for i, j in img:`` slower than:
    for i, j in ti.ndrange(*img.shape):
        r, g, b = 0, 0, 0
        color = img[i, img.shape[1] - 1 - j]
        if ti.static(img.dtype in [ti.f32, ti.f64]):
            r, g, b = min(255, max(0, int(color * 255)))
        else:
            impl.static_assert(img.dtype == ti.u8)
            r, g, b = color
        idx = j * img.shape[0] + i
        # We use i32 for |out| since OpenGL and Metal doesn't support u8 types
        if ti.static(settings.get_os_name() != 'osx'):
            out[idx] = (r << 16) + (g << 8) + b
        else:
            # What's -16777216?
            #
            # On Mac, we need to set the alpha channel to 0xff. Since Mac's GUI
            # is big-endian, the color is stored in ABGR order, and we need to
            # add 0xff000000, which is -16777216 in I32's legit range. (Albeit
            # the clarity, adding 0xff000000 doesn't work.)
            alpha = -16777216
            out[idx] = (b << 16) + (g << 8) + r + alpha


@kernel
def tensor_to_image(tensor: template(), arr: ext_arr()):
    for I in ti.grouped(tensor):
        t = ti.cast(tensor[I], ti.f32)
        arr[I, 0] = t
        arr[I, 1] = t
        arr[I, 2] = t


@kernel
def vector_to_image(mat: template(), arr: ext_arr()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            arr[I, p] = ti.cast(mat[I][p], ti.f32)
            if ti.static(mat.n <= 2):
                arr[I, 2] = 0


@kernel
def tensor_to_tensor(tensor: template(), other: template()):
    for I in ti.grouped(tensor):
        tensor[I] = other[I]


@kernel
def ext_arr_to_tensor(arr: ext_arr(), tensor: template()):
    for I in ti.grouped(tensor):
        tensor[I] = arr[I]


@kernel
def matrix_to_ext_arr(mat: template(), arr: ext_arr(), as_vector: template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                if ti.static(as_vector):
                    arr[I, p] = mat[I][p]
                else:
                    arr[I, p, q] = mat[I][p, q]


@kernel
def ext_arr_to_matrix(arr: ext_arr(), mat: template(), as_vector: template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                if ti.static(as_vector):
                    mat[I][p] = arr[I, p]
                else:
                    mat[I][p, q] = arr[I, p, q]


@kernel
def clear_gradients(vars: template()):
    for I in ti.grouped(ScalarField(Expr(vars[0]))):
        for s in ti.static(vars):
            ScalarField(Expr(s))[I] = 0


@kernel
def clear_loss(l: template()):
    # Using SNode writers would result in a forced sync, therefore we wrap these
    # writes into a kernel.
    l[None] = 0
    l.grad[None] = 1


@kernel
def fill_matrix(mat: template(), vals: template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                mat[I][p, q] = vals[p][q]


@kernel
def snode_deactivate(b: template()):
    for I in ti.grouped(b):
        ti.deactivate(b, I)


@kernel
def snode_deactivate_dynamic(b: template()):
    for I in ti.grouped(b.parent()):
        ti.deactivate(b, I)
