import taichi as ti
import math
from struct import pack, unpack

ti.init()

cft = ti.quant.float(exp=6, frac=13, signed=False)
# cft = ti.quant.fixed(frac=16, range=1024)
a = ti.field(dtype=cft)
b = ti.field(dtype=cft)
shared_exp = ti.root.bit_struct(num_bits=32)
shared_exp.place(a, b, shared_exponent=True)
# s.place(a, b)


@ti.kernel
def set_vals(x: ti.f32, y: ti.f32):
    a[None] = x
    b[None] = y


def serialize_i32(x):
    s = ''
    for i in reversed(range(32)):
        s += f'{(x>>i) & 1}'
    return s

def serialize_f32(x):
    b = pack('f', x)
    n = unpack('i', b)[0]
    return serialize_i32(n)


@ti.kernel
def fetch_shared_exp() -> ti.i32:
    return shared_exp[None]


coord = ti.GUI(res=(800, 800), background_color=0xFFFFFF)
numbers = ti.GUI(res=(800, 800), background_color=0xFFFFFF)


def draw_coord(t, f):
    cx, cy = 0.5, 0.5
    lx, ly = 0.4, 0.4
    l1 = lx * 0.8
    al = 0.02
    coord.line(begin=(cx - lx, cy),
               end=(cx + lx, cy),
               radius=3,
               color=0x666666)
    coord.line(begin=(cx, cy - ly),
               end=(cx, cy + ly),
               radius=3,
               color=0x666666)
    coord.line(begin=(cx + lx - al, cy - al),
               end=(cx + lx, cy),
               radius=3,
               color=0x666666)
    coord.line(begin=(cx + lx - al, cy + al),
               end=(cx + lx, cy),
               radius=3,
               color=0x666666)
    coord.line(begin=(cx - al, cy + ly - al),
               end=(cx, cy + ly),
               radius=3,
               color=0x666666)
    coord.line(begin=(cx + al, cy + ly - al),
               end=(cx, cy + ly),
               radius=3,
               color=0x666666)

    def transform(p):
        return cx + l1 * p[0], cy + l1 * p[1]

    segments = 300
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        coord.line(begin=transform(f(t1)),
                   end=transform(f(t2)),
                   radius=3,
                   color=0x0)

    coord.circle(pos=transform(f(t)), color=0xDD1122, radius=10)


frames = 1000


def f(t):
    return 1 - t, t


for i in range(frames * 1000):
    t = (i % frames) / (frames - 1)
    t = math.sin(t * math.pi * 2) * 0.5 + 0.5

    draw_coord(t, f)
    coord.show()

    x, y = f(t)
    set_vals(x, y)
    se = serialize_i32(fetch_shared_exp())

    fs = 100
    color = 0x111111
    numbers.text(f'{x:.5f}', (0.05, 0.9), font_size=fs, color=color)
    numbers.text(f'{y:.5f}', (0.55, 0.9), font_size=fs, color=color)
    
    fs = 49
    
    # Change the order for more clarity
    se = se[-6:] + se[-19:-6] + se[:-19]
    numbers.text(se, (0.05, 0.7), font_size=fs, color=color)

    numbers.show()
