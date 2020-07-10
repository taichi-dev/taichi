import taichi as ti
import os
import ctypes

ti.init()

os.system("g++ a.cpp -o a.so -fPIC -shared")

so = ctypes.CDLL("./a.so")

@ti.kernel
def call_ext():
    a = 2.0
    b = 3.0
    c = 0.0
    d = 0.0
    e = 3
    ti.external_func_call(func=so.add_and_mul, args=(a, b), outputs=(c, d, e))
    print(c, d, e)
    
call_ext()
