import taichi as ti
import os
import ctypes

os.system("g++ a.cpp -o a.so -fPIC -shared")

so = ctypes.CDLL("./a.so")
func = so.add
func.argtypes = [ctypes.c_float, ctypes.c_float]
func.restype = ctypes.c_float

print(func(1, 2))

@ti.kernel
def call():
    ti.external_func_call()
