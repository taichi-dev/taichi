import argparse
import math
import os
from struct import pack, unpack

import taichi as ti

ti.init()

f19 = ti.types.quant.float(exp=6, frac=13, signed=True)
f16 = ti.types.quant.float(exp=5, frac=11, signed=True)
fixed16 = ti.types.quant.fixed(frac=16, range=2)

vf19 = ti.Vector.field(2, dtype=f19)
bs_vf19 = ti.root.bit_struct(num_bits=32)
bs_vf19.place(vf19, shared_exponent=True)

vf16 = ti.Vector.field(2, dtype=f16)
bs_vf16 = ti.root.bit_struct(num_bits=32)
bs_vf16.place(vf16)

vfixed16 = ti.Vector.field(2, dtype=fixed16)
bs_vfixed16 = ti.root.bit_struct(num_bits=32)
bs_vfixed16.place(vfixed16)


@ti.kernel
def set_vals(x: ti.f32, y: ti.f32):
    val = ti.Vector([x, y])
    vf16[None] = val
    vf19[None] = val
    vfixed16[None] = val


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
def fetch_bs(bs: ti.template()) -> ti.i32:
    return bs[None]


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


frames = 300

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--curve', type=int, help='Curve type', default=0)

args = parser.parse_args()

if args.curve == 0:

    def f(t):
        return math.cos(t * 2 * math.pi), math.sin(t * 2 * math.pi)
elif args.curve == 1:

    def f(t):
        t = math.cos(t * 2 * math.pi) * 0.5 + 0.5
        return 1 - t, t
elif args.curve == 2:

    def f(t):
        t = math.cos(t * 2 * math.pi)
        t = t * 2.3
        s = 0.1
        return math.exp(t) * s, math.exp(-t) * s


folder = f'curve{args.curve}'
os.makedirs(folder, exist_ok=True)

for i in range(frames * 2 + 1):
    t = i / frames

    draw_coord(t, f)
    coord.show(f'{folder}/coord_{i:04d}.png')

    x, y = f(t)
    set_vals(x, y)

    fs = 100
    color = 0x111111

    def reorder(b, seg):
        r = ''
        seg = [0] + seg + [32]
        for i in range(len(seg) - 1):
            r = r + b[32 - seg[i + 1]:32 - seg[i]]
        return r

    def real_to_str(x):
        s = ''
        if x < 0:
            s = ''
        else:
            s = ' '
        return s + f'{x:.4f}'

    numbers.text(real_to_str(x), (0.05, 0.9), font_size=fs, color=color)
    numbers.text(real_to_str(y), (0.55, 0.9), font_size=fs, color=color)

    fs = 49

    bits = [bs_vf19, bs_vf16, bs_vfixed16]
    seg = [[], [], [6, 19], [5, 16, 21], [16]]
    bits = list(map(lambda x: serialize_i32(fetch_bs(x)), bits))

    bits = [serialize_f32(x), serialize_f32(y)] + bits

    for j in range(len(bits)):
        b = reorder(bits[j], seg[j])
        numbers.text(b, (0.05, 0.7 - j * 0.15), font_size=fs, color=color)

    numbers.show(f'{folder}/numbers_{i:04d}.png')

os.system(
    f'ti video {folder}/numbers*.png -f 60 -c 2 -o numbers{args.curve}.mp4')
os.system(f'ti video {folder}/coord*.png -f 60 -c 2 -o coord{args.curve}.mp4')
