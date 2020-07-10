import taichi as ti
import os
import ctypes

os.system("g++ a.cpp -o a.so -fPIC -shared")

so = ctypes.CDLL("./a.so")
func = so.add

print(func(1, 2))

@ti.kernel
def call():
    ti.external_func_call()
