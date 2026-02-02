"""
Taichi FlagOS Backend Example: Julia Set Fractal

This example demonstrates how to use the FlagOS backend in Taichi
to run computations on various AI chips (MLU, Ascend, DCU, etc.)

Prerequisites:
1. Install FlagOS SDK and FlagTree compiler
2. Build Taichi with -DTI_WITH_FLAGOS=ON
3. Set environment variable: export TI_FLAGOS_CHIP=mlu370

Supported chips:
- mlu370, mlu590 (Cambricon)
- ascend910, ascend310 (Huawei)
- dcu (Hygon)
- gcu (Enflame)
- generic (fallback)
"""

import taichi as ti
import os

# Check if FlagOS backend is available
try:
    ti.flagos
    FLAGOS_AVAILABLE = True
except AttributeError:
    FLAGOS_AVAILABLE = False
    print("Warning: FlagOS backend not available. Falling back to CPU.")

# Initialize Taichi with FlagOS backend
if FLAGOS_AVAILABLE:
    # Method 1: Use environment variable
    # export TI_FLAGOS_CHIP=mlu370

    # Method 2: Specify chip in init
    ti.init(arch=ti.flagos, flagos_chip="mlu370")

    print(f"Using FlagOS backend with chip: {ti.cfg.flagos_chip}")
else:
    ti.init(arch=ti.cpu)

# Parameters
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    """Complex number square"""
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    """Julia set computation kernel"""
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


def main():
    """Main function"""
    print("Taichi FlagOS Backend Demo - Julia Set Fractal")
    print("=" * 50)

    # Warmup
    print("Warming up...")
    paint(0.0)
    ti.sync()

    # Benchmark
    import time

    print("Running benchmark...")
    start = time.time()

    iterations = 100
    for i in range(iterations):
        paint(i * 0.03)

    ti.sync()
    end = time.time()

    print(f"Time for {iterations} iterations: {end - start:.3f}s")
    print(f"Average time per iteration: {(end - start) / iterations * 1000:.3f}ms")

    # Visualization (only if GUI is available)
    try:
        gui = ti.GUI("Julia Set (FlagOS)", res=(n * 2, n))
        for i in range(1000000):
            paint(i * 0.03)
            gui.set_image(pixels)
            gui.show()
    except Exception as e:
        print(f"GUI not available: {e}")
        print("Computation completed successfully!")


if __name__ == "__main__":
    main()
