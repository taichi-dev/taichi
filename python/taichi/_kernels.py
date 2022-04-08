from taichi._lib.utils import get_os_name
from taichi.lang import ops
from taichi.lang._ndrange import ndrange
from taichi.lang.expr import Expr
from taichi.lang.field import ScalarField
from taichi.lang.impl import grouped, static, static_assert
from taichi.lang.kernel_impl import kernel
from taichi.lang.runtime_ops import sync
from taichi.lang.snode import deactivate
from taichi.types import ndarray_type
from taichi.types.annotations import template
from taichi.types.primitive_types import f16, f32, f64, u8


# A set of helper (meta)functions
@kernel
def fill_tensor(tensor: template(), val: template()):
    for I in grouped(tensor):
        tensor[I] = val


@kernel
def fill_ndarray(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I] = val


@kernel
def fill_ndarray_matrix(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I].fill(val)


@kernel
def tensor_to_ext_arr(tensor: template(), arr: ndarray_type.ndarray()):
    for I in grouped(tensor):
        arr[I] = tensor[I]


@kernel
def ndarray_to_ext_arr(ndarray: ndarray_type.ndarray(),
                       arr: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        arr[I] = ndarray[I]


@kernel
def ndarray_matrix_to_ext_arr(ndarray: ndarray_type.ndarray(),
                              arr: ndarray_type.ndarray(),
                              layout_is_aos: template(),
                              as_vector: template()):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            for q in static(range(ndarray[I].m)):
                if static(as_vector):
                    if static(layout_is_aos):
                        arr[I, p] = ndarray[I][p]
                    else:
                        arr[p, I] = ndarray[I][p]
                else:
                    if static(layout_is_aos):
                        arr[I, p, q] = ndarray[I][p, q]
                    else:
                        arr[p, q, I] = ndarray[I][p, q]


@kernel
def vector_to_fast_image(img: template(), out: ndarray_type.ndarray()):
    # FIXME: Why is ``for i, j in img:`` slower than:
    for i, j in ndrange(*img.shape):
        r, g, b = 0, 0, 0
        color = img[i, img.shape[1] - 1 - j]
        if static(img.dtype in [f16, f32, f64]):
            r, g, b = min(255, max(0, int(color * 255)))
        else:
            static_assert(img.dtype == u8)
            r, g, b = color
        idx = j * img.shape[0] + i
        # We use i32 for |out| since OpenGL and Metal doesn't support u8 types
        if static(get_os_name() != 'osx'):
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
def tensor_to_image(tensor: template(), arr: ndarray_type.ndarray()):
    for I in grouped(tensor):
        t = ops.cast(tensor[I], f32)
        arr[I, 0] = t
        arr[I, 1] = t
        arr[I, 2] = t


@kernel
def vector_to_image(mat: template(), arr: ndarray_type.ndarray()):
    for I in grouped(mat):
        for p in static(range(mat.n)):
            arr[I, p] = ops.cast(mat[I][p], f32)
            if static(mat.n <= 2):
                arr[I, 2] = 0


@kernel
def tensor_to_tensor(tensor: template(), other: template()):
    for I in grouped(tensor):
        tensor[I] = other[I]


@kernel
def ext_arr_to_tensor(arr: ndarray_type.ndarray(), tensor: template()):
    for I in grouped(tensor):
        tensor[I] = arr[I]


@kernel
def ndarray_to_ndarray(ndarray: ndarray_type.ndarray(),
                       other: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = other[I]


@kernel
def ext_arr_to_ndarray(arr: ndarray_type.ndarray(),
                       ndarray: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = arr[I]


@kernel
def ext_arr_to_ndarray_matrix(arr: ndarray_type.ndarray(),
                              ndarray: ndarray_type.ndarray(),
                              layout_is_aos: template(),
                              as_vector: template()):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            for q in static(range(ndarray[I].m)):
                if static(as_vector):
                    if static(layout_is_aos):
                        ndarray[I][p] = arr[I, p]
                    else:
                        ndarray[I][p] = arr[p, I]
                else:
                    if static(layout_is_aos):
                        ndarray[I][p, q] = arr[I, p, q]
                    else:
                        ndarray[I][p, q] = arr[p, q, I]


@kernel
def matrix_to_ext_arr(mat: template(), arr: ndarray_type.ndarray(),
                      as_vector: template()):
    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(as_vector):
                    arr[I, p] = mat[I][p]
                else:
                    arr[I, p, q] = mat[I][p, q]


@kernel
def ext_arr_to_matrix(arr: ndarray_type.ndarray(), mat: template(),
                      as_vector: template()):
    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(as_vector):
                    mat[I][p] = arr[I, p]
                else:
                    mat[I][p, q] = arr[I, p, q]


@kernel
def clear_gradients(_vars: template()):
    for I in grouped(ScalarField(Expr(_vars[0]))):
        for s in static(_vars):
            ScalarField(Expr(s))[I] = 0


@kernel
def clear_loss(l: template()):
    # Using SNode writers would result in a forced sync, therefore we wrap these
    # writes into a kernel.
    l[None] = 0
    l.grad[None] = 1


@kernel
def fill_matrix(mat: template(), vals: template()):
    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                mat[I][p, q] = vals[p][q]


@kernel
def snode_deactivate(b: template()):
    for I in grouped(b):
        deactivate(b, I)


@kernel
def snode_deactivate_dynamic(b: template()):
    for I in grouped(b.parent()):
        deactivate(b, I)


# Odd-even merge sort
# References:
# https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
# https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
@kernel
def sort_stage(keys: template(), use_values: int, values: template(), N: int,
               p: int, k: int, invocations: int):
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ops.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a
                    if use_values != 0:
                        temp = values[a]
                        values[a] = values[b]
                        values[b] = temp


def parallel_sort(keys, values=None):
    N = keys.shape[0]

    num_stages = 0
    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            if values is None:
                sort_stage(keys, 0, keys, N, p, k, invocations)
            else:
                sort_stage(keys, 1, values, N, p, k, invocations)
            num_stages += 1
            sync()
            k = int(k / 2)
        p = int(p * 2)
    print(num_stages)
