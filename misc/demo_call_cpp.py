import ctypes
import os

import taichi as ti

ti.init(dynamic_index=True)

source = '''
void foo(int* a, int b[2][2]) {
    a[0] = 11;
    b[0][0] = 10;
}
'''

with open('a.c', 'w') as f:
    f.write(source)

# TODO: Integrate this setup into taichi
os.system("clang-11 -flto -c a.c -o a.bc")

@ti.kernel
def call_ext():
    b = 0
    a = ti.Vector([[0,0],[0,0]], ti.i32)
    print(a, b)
    ti.call_cpp("a.bc", "foo", b, a)
    print(a, b)

call_ext()


os.remove('a.c')
os.remove('a.bc')
