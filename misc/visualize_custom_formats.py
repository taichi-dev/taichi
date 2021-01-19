import taichi as ti
from struct import pack, unpack

ti.init()

cft = ti.quant.float(exp=6, frac=13)
# cft = ti.quant.fixed(frac=16, range=1024)
a = ti.field(dtype=cft)
b = ti.field(dtype=cft)
s = ti.root.bit_struct(num_bits=32)
s.place(a, b, shared_exponent=True)
# s.place(a, b)


@ti.kernel
def set_vals(x: ti.f32, y: ti.f32):
    a[None] = x
    b[None] = y


def print_i32(x, splits=[]):
    for i in reversed(range(32)):
        print(f'{(x>>i) & 1}', end='')
        if i in splits:
            print(' ', end='')
    print()


def print_f32(x):
    b = pack('f', x)
    n = unpack('i', b)[0]
    print_i32(n, [31, 23])


@ti.kernel
def fetch() -> ti.i32:
    return s[None]


a[None] = 0.5
b[None] = 0.25

x = 2**7 * 1.25
y = 2**-7 * 1.25

for i in range(40):
    set_vals(x, y)
    print('values:')
    print(x, a[None], y, b[None])
    print('bit_struct:')
    print_i32(fetch(), [6, 19])
    print()

    x = x / 2
    y = y * 2
