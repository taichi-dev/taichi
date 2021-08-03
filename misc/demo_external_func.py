import ctypes
import os

import taichi as ti

ti.init()

N = 1024
x = ti.field(ti.i32, shape=N)
y = ti.field(ti.i32, shape=N)
z = ti.field(ti.i32, shape=N)

source = '''
extern "C" {
    void add_and_mul(float a, float b, float *c, float *d, int *e) {
        *c = a + b;
        *d = a * b;
        *e = int(a * b + a);
    }
    void pow_int(int a, int b, int *c) {
        int ret = 1;
        for (int i = 0; i < b; i++)
            ret = ret * a;
        *c = ret;
    }
}
'''

with open('a.cpp', 'w') as f:
    f.write(source)

os.system("g++ a.cpp -o a.so -fPIC -shared")

so = ctypes.CDLL("./a.so")


@ti.kernel
def call_ext() -> ti.i32:
    a = 2.0
    b = 3.0
    c = 0.0
    d = 0.0
    e = 3
    ti.external_func_call(func=so.add_and_mul, args=(a, b), outputs=(c, d, e))
    p = 0
    ti.external_func_call(func=so.pow_int, args=(int(c + d), e), outputs=(p, ))
    return p


# Wrap the external function to make it easier to use
@ti.func
def pow_int_wrapper(a, b):
    p = 0
    ti.external_func_call(func=so.pow_int,
                          args=(int(a), int(b)),
                          outputs=(p, ))
    return p


@ti.kernel
def call_parallel():
    for i in range(N):
        z[i] = pow_int_wrapper(x[i], y[i])


assert call_ext() == 11**8

for i in range(N):
    x[i] = i
    y[i] = 3

call_parallel()
for i in range(N):
    assert z[i] == i**3

os.remove('a.cpp')
os.remove('a.so')
