import argparse
import os

import taichi as ti


def compile_numerical_aot_test(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def fill_scalar_array_with_fp32(arr: ti.types.ndarray(dtype=ti.f16, ndim=1), val: ti.f32):
        for i in arr:
            arr[i] = ti.cast(val, ti.f16)

    @ti.kernel
    def fill_scalar_array_with_fp16(arr: ti.types.ndarray(dtype=ti.f16, ndim=1), val: ti.f16):
        for i in arr:
            arr[i] = val

    @ti.kernel
    def fill_matrix_array_with_fp16(
        arr: ti.types.ndarray(dtype=ti.types.matrix(n=2, m=3, dtype=ti.f16), ndim=1),
        val: ti.f16,
    ):
        for i in arr:
            arr[i] = [[val, val, val], [val, val, val]]

    @ti.kernel
    def compute_kernel(
        pose: ti.types.ndarray(dtype=ti.types.matrix(3, 4, dtype=ti.f16), ndim=0),
        directions: ti.types.ndarray(dtype=ti.types.matrix(1, 3, dtype=ti.f16), ndim=1),
        out_0: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f16), ndim=1),
        out_1: ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f16), ndim=1),
    ):
        for i in range(directions.shape[0]):
            c2w = pose[None]
            mat_result = directions[i] @ c2w[:, :3].transpose()
            ray_d = ti.Vector([mat_result[0, 0], mat_result[0, 1], mat_result[0, 2]])
            ray_o = c2w[:, 3]

            out_0[i] = ray_o
            out_1[i] = ray_d

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(caps=["spirv_has_int16", "spirv_has_float16"])
    m.add_kernel(fill_scalar_array_with_fp32)
    m.add_kernel(fill_scalar_array_with_fp16)
    m.add_kernel(fill_matrix_array_with_fp16)
    m.add_kernel(compute_kernel)
    m.save(dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cpu":
        compile_numerical_aot_test(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_numerical_aot_test(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_numerical_aot_test(arch=ti.vulkan)
    elif args.arch == "metal":
        compile_numerical_aot_test(arch=ti.metal)
    elif args.arch == "opengl":
        compile_numerical_aot_test(arch=ti.opengl)
    elif args.arch == "dx12":
        compile_numerical_aot_test(arch=ti.dx12)
    else:
        assert False
