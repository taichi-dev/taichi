import taichi as ti
import time

t = time.time()
ti.init(arch=ti.cuda, print_kernel_llvm_ir_optimized=True)


@ti.kernel
def p():
    print(42)


p()

print(f'{time.time() - t:.3f} s')
ti.core.print_profile_info()