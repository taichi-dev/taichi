from taichi.core import settings
from taichi.lang import impl
from taichi.lang.expr import Expr

import taichi as ti

# A set of helper (meta)functions


@ti.kernel
def fill_tensor(tensor: ti.template(), val: ti.template()):
    for I in ti.grouped(tensor):
        tensor[I] = val


@ti.kernel
def tensor_to_ext_arr(tensor: ti.template(), arr: ti.ext_arr()):
    for I in ti.grouped(tensor):
        arr[I] = tensor[I]


@ti.kernel
def vector_to_fast_image(img: ti.template(), out: ti.ext_arr()):
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


@ti.kernel
def tensor_to_image(tensor: ti.template(), arr: ti.ext_arr()):
    for I in ti.grouped(tensor):
        t = ti.cast(tensor[I], ti.f32)
        arr[I, 0] = t
        arr[I, 1] = t
        arr[I, 2] = t


@ti.kernel
def vector_to_image(mat: ti.template(), arr: ti.ext_arr()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            arr[I, p] = ti.cast(mat[I][p], ti.f32)
            if ti.static(mat.n <= 2):
                arr[I, 2] = 0


@ti.kernel
def tensor_to_tensor(tensor: ti.template(), other: ti.template()):
    for I in ti.grouped(tensor):
        tensor[I] = other[I]


@ti.kernel
def ext_arr_to_tensor(arr: ti.ext_arr(), tensor: ti.template()):
    for I in ti.grouped(tensor):
        tensor[I] = arr[I]


@ti.kernel
def matrix_to_ext_arr(mat: ti.template(), arr: ti.ext_arr(),
                      as_vector: ti.template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                if ti.static(as_vector):
                    arr[I, p] = mat[I][p]
                else:
                    arr[I, p, q] = mat[I][p, q]


@ti.kernel
def ext_arr_to_matrix(arr: ti.ext_arr(), mat: ti.template(),
                      as_vector: ti.template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                if ti.static(as_vector):
                    mat[I][p] = arr[I, p]
                else:
                    mat[I][p, q] = arr[I, p, q]


@ti.kernel
def clear_gradients(vars: ti.template()):
    for I in ti.grouped(Expr(vars[0])):
        for s in ti.static(vars):
            Expr(s)[I] = 0


@ti.kernel
def clear_loss(l: ti.template()):
    # Using SNode writers would result in a forced sync, therefore we wrap these
    # writes into a kernel.
    l[None] = 0
    l.grad[None] = 1


@ti.kernel
def fill_matrix(mat: ti.template(), vals: ti.template()):
    for I in ti.grouped(mat):
        for p in ti.static(range(mat.n)):
            for q in ti.static(range(mat.m)):
                mat[I][p, q] = vals[p][q]


@ti.kernel
def snode_deactivate(b: ti.template()):
    for I in ti.grouped(b):
        ti.deactivate(b, I)


@ti.kernel
def snode_deactivate_dynamic(b: ti.template()):
    for I in ti.grouped(b.parent()):
        ti.deactivate(b, I)
