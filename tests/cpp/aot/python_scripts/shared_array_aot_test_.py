import argparse
import os

import numpy as np

import taichi as ti


def shared_array_aot_test(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return
    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_arr = np.zeros(N).astype(np.float32)
    d_arr = np.zeros(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)

    @ti.kernel
    def run(
        v: ti.types.ndarray(ndim=1),
        d: ti.types.ndarray(ndim=1),
        a: ti.types.ndarray(ndim=1),
    ):
        for i in range(nBlocks * block_dim):
            v[i] = 1.0
            d[i] = 1.0

        ti.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = ti.simt.block.SharedArray((block_dim,), ti.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                ti.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                ti.simt.block.sync()
            a[i] = acc

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module()
    m.add_kernel(run, template_args={"v": v_arr, "d": d_arr, "a": a_arr})
    m.save(dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cuda":
        shared_array_aot_test(arch=ti.cuda)
    elif args.arch == "vulkan":
        shared_array_aot_test(arch=ti.vulkan)
    elif args.arch == "metal":
        shared_array_aot_test(arch=ti.metal)
    else:
        assert False
