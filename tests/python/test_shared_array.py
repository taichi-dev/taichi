import numpy as np

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.amdgpu])
def test_shared_array_nested_loop():
    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @ti.kernel
    def calc(v: ti.types.ndarray(ndim=1), d: ti.types.ndarray(ndim=1),
             a: ti.types.ndarray(ndim=1)):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @ti.kernel
    def calc_shared_array(v: ti.types.ndarray(ndim=1),
                          d: ti.types.ndarray(ndim=1),
                          a: ti.types.ndarray(ndim=1)):
        ti.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = ti.simt.block.SharedArray((block_dim, ), ti.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                ti.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                ti.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.amdgpu])
def test_shared_array_atomics():
    N = 256
    block_dim = 32

    @ti.kernel
    def atomic_test(out: ti.types.ndarray()):
        ti.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = tid
            sharr = ti.simt.block.SharedArray((block_dim, ), ti.i32)
            sharr[tid] = val
            ti.simt.block.sync()
            sharr[0] += val
            ti.simt.block.sync()
            out[i] = sharr[tid]

    arr = ti.ndarray(ti.i32, (N))
    atomic_test(arr)
    ti.sync()
    sum = block_dim * (block_dim - 1) // 2
    assert arr[0] == sum
    assert arr[32] == sum
    assert arr[128] == sum
    assert arr[224] == sum
