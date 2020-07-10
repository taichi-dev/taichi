import taichi as ti
import os
import ctypes

ti.init(print_ir=True, print_kernel_llvm_ir=True)

os.system("g++ a.cpp -o a.so -fPIC -shared")

so = ctypes.CDLL("./a.so")
func = so.add
func.argtypes = [ctypes.c_float, ctypes.c_float]
func.restype = ctypes.c_float
print(dir(func))

print(func(1, 2))

add_and_mul = so.add_and_mul

addr = ctypes.addressof(add_and_mul)

@ti.kernel
def call():
    a = 2.0
    b = 3.0
    c = 0.0
    d = 0.0
    ti.external_func_call(addr, (a, b), (c, d))
    print(c, d)
    
call()
