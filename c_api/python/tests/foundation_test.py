import taichi_runtime as ti
import numpy as np
from pathlib import Path

TEST_DIR = str(Path(__file__).parent)


def test_create_destroy_runtime():
    runtime = ti.Runtime.create(arch=[ti.Arch.METAL, ti.Arch.VULKAN])
    runtime.destroy()


def test_allocate_free_ndarray():
    runtime = ti.Runtime.create(arch=[ti.Arch.METAL, ti.Arch.VULKAN])
    x = ti.NdArray.allocate(runtime, ti.DataType.I32, shape=[4], elem_shape=[])
    x.free()
    runtime.destroy()


def test_numpy_interop():
    runtime = ti.Runtime.create(arch=[ti.Arch.METAL, ti.Arch.VULKAN])
    x = np.arange(5 * 4 * 3).reshape(5, 4, 3).astype(np.int32)
    y = ti.NdArray.from_numpy(runtime, x, elem_shape=(3,))
    z = y.into_numpy()
    assert np.allclose(x, z)
    runtime.destroy()


def test_kernel_launch():
    runtime = ti.Runtime.create(arch=[ti.Arch.METAL, ti.Arch.VULKAN])
    module = ti.AotModule.load(runtime, path=TEST_DIR + "/assets/arange.py.tcm")
    kernel = module.get_kernel("arange")
    x = ti.NdArray.allocate(runtime, ti.DataType.I32, shape=[4], elem_shape=[])
    kernel.launch(x)
    runtime.wait()
    assert np.allclose(x.into_numpy(), np.arange(4))
    runtime.destroy()
