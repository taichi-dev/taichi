from taichi._lib.utils import get_os_name
from taichi.lang import ops
from taichi.lang._ndrange import ndrange
from taichi.lang.expr import Expr
from taichi.lang.field import ScalarField
from taichi.lang.impl import current_cfg, field, grouped, static, static_assert
from taichi.lang.kernel_impl import func, kernel
from taichi.lang.misc import cuda, loop_config, vulkan
from taichi.lang.runtime_ops import sync
from taichi.lang.simt import block, subgroup, warp
from taichi.lang.snode import deactivate
from taichi.types import ndarray_type, texture_type, vector
from taichi.types.annotations import template
from taichi.types.primitive_types import f16, f32, f64, i32, u8


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
            r, g, b = min(255, max(0, int(color * 255)))[:3]
        else:
            static_assert(img.dtype == u8)
            r, g, b = color[:3]

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
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I, p] = mat[I][p]
                    else:
                        arr[I, p] = mat[I][p, q]
                else:
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I, p, q] = mat[I][p]
                    else:
                        arr[I, p, q] = mat[I][p, q]


@kernel
def ext_arr_to_matrix(arr: ndarray_type.ndarray(), mat: template(),
                      as_vector: template()):
    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(getattr(mat, "ndim", 2) == 1):
                    if static(as_vector):
                        mat[I][p] = arr[I, p]
                    else:
                        mat[I][p] = arr[I, p, q]
                else:
                    if static(as_vector):
                        mat[I][p, q] = arr[I, p]
                    else:
                        mat[I][p, q] = arr[I, p, q]


# extract ndarray of raw vulkan memory layout to normal memory layout.
# the vulkan layout stored in ndarray : width-by-width stored along n-
# darray's shape[1] which is the height-axis(So use [size // h, size %
#  h]). And the height-order of vulkan layout is flip up-down.(So take
# [size = (h - 1 - j) * w + i] to get the index)
@kernel
def arr_vulkan_layout_to_arr_normal_layout(vk_arr: ndarray_type.ndarray(),
                                           normal_arr: ndarray_type.ndarray()):
    static_assert(len(normal_arr.shape) == 2)
    w = normal_arr.shape[0]
    h = normal_arr.shape[1]
    for i, j in ndrange(w, h):
        normal_arr[i, j] = vk_arr[(h - 1 - j) * w + i]


# extract ndarray of raw vulkan memory layout into a taichi-field data
# structure with normal memory layout.
@kernel
def arr_vulkan_layout_to_field_normal_layout(vk_arr: ndarray_type.ndarray(),
                                             normal_field: template()):
    static_assert(len(normal_field.shape) == 2)
    w = normal_field.shape[0]
    h = normal_field.shape[1]
    for i, j in ndrange(w, h):
        normal_field[i, j] = vk_arr[(h - 1 - j) * w + i]


@kernel
def clear_gradients(_vars: template()):
    for I in grouped(ScalarField(Expr(_vars[0]))):
        for s in static(_vars):
            ScalarField(Expr(s))[I] = ops.cast(0, dtype=s.get_dt())


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
                if static(mat[I].ndim == 2):
                    mat[I][p, q] = vals[p][q]
                else:
                    mat[I][p] = vals[p][q]


@kernel
def snode_deactivate(b: template()):
    for I in grouped(b):
        deactivate(b, I)


@kernel
def snode_deactivate_dynamic(b: template()):
    for I in grouped(b.parent()):
        deactivate(b, I)


@kernel
def load_texture_from_numpy(tex: texture_type.rw_texture(num_dimensions=2,
                                                         num_channels=4,
                                                         channel_format=u8,
                                                         lod=0),
                            img: ndarray_type.ndarray(field_dim=2,
                                                      element_shape=(3, ))):
    for i, j in img:
        tex.store(
            vector(2, i32)([i, j]),
            vector(4, f32)([img[i, j][0], img[i, j][1], img[i, j][2], 0]) /
            255.)


@kernel
def save_texture_to_numpy(tex: texture_type.rw_texture(num_dimensions=2,
                                                       num_channels=4,
                                                       channel_format=u8,
                                                       lod=0),
                          img: ndarray_type.ndarray(field_dim=2,
                                                    element_shape=(3, ))):
    for i, j in img:
        img[i, j] = ops.round(tex.load(vector(2, i32)([i, j])).rgb * 255)


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


@func
def warp_shfl_up_i32(val: template()):
    global_tid = block.global_thread_idx()
    WARP_SZ = 32
    lane_id = global_tid % WARP_SZ
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@kernel
def scan_add_inclusive(arr_in: template(), in_beg: i32, in_end: i32,
                       single_block: template(), inclusive_add: template()):
    WARP_SZ = 32
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = block.SharedArray((65, ), i32)

        val = inclusive_add(val)
        block.sync()

        # Put warp scan results to smem
        # TODO replace smem with real smem when available
        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val
        block.sync()

        # Inter-warp scan, use the first thread in the first warp
        if warp_id == 0 and lane_id == 0:
            for k in range(1, BLOCK_SZ / WARP_SZ):
                pad_shared[k] += pad_shared[k - 1]
        block.sync()

        # Update data with warp sums
        warp_sum = 0
        if warp_id > 0:
            warp_sum = pad_shared[warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        # Update partial sums except the final block
        if not single_block and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@kernel
def uniform_add(arr_in: template(), in_beg: i32, in_end: i32):
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


@kernel
def blit_from_field_to_field(
        dst: template(), src: template(), offset: i32, size: i32):
    for i in range(size):
        dst[i + offset] = src[i]


# Parallel Prefix Sum (Scan)
# Ref[0]: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
# Ref[1]: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
def prefix_sum_inclusive_inplace(input_arr, length):
    BLOCK_SZ = 64
    GRID_SZ = int((length + BLOCK_SZ - 1) / BLOCK_SZ)

    # Buffer position and length
    # This is a single buffer implementation for ease of aot usage
    ele_num = length
    ele_nums = [ele_num]
    start_pos = 0
    ele_nums_pos = [start_pos]

    while ele_num > 1:
        ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
        ele_nums.append(ele_num)
        start_pos += BLOCK_SZ * ele_num
        ele_nums_pos.append(start_pos)

    if input_arr.dtype != i32:
        raise RuntimeError("Only ti.i32 type is supported for prefix sum.")

    large_arr = field(i32, shape=start_pos)

    if current_cfg().arch == cuda:
        inclusive_add = warp_shfl_up_i32
    elif current_cfg().arch == vulkan:
        inclusive_add = subgroup.inclusive_add
    else:
        raise RuntimeError(
            f"{str(current_cfg().arch)} is not supported for prefix sum.")

    blit_from_field_to_field(large_arr, input_arr, 0, length)

    # Kogge-Stone construction
    for i in range(len(ele_nums) - 1):
        if i == len(ele_nums) - 2:
            scan_add_inclusive(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1],
                               True, inclusive_add)
        else:
            scan_add_inclusive(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1],
                               False, inclusive_add)

    for i in range(len(ele_nums) - 3, -1, -1):
        uniform_add(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

    blit_from_field_to_field(input_arr, large_arr, 0, length)
