"""
Use 4D simplex noise to reproduce a generative artwork by https://bleuje.com/
"""
import numpy as np

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

F4 = 0.30901699437494745
G4 = 0.1381966011250105

grad4_np = np.array(
    [[0, 1, 1, 1], [0, 1, 1, -1], [0, 1, -1, 1], [0, 1, -1, -1], [0, -1, 1, 1],
     [0, -1, 1, -1], [0, -1, -1, 1], [0, -1, -1, -1], [1, 0, 1, 1],
     [1, 0, 1, -1], [1, 0, -1, 1], [1, 0, -1, -1],
     [-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, -1, 1], [-1, 0, -1, -1],
     [1, 1, 0, 1], [1, 1, 0, -1], [1, -1, 0, 1], [1, -1, 0, -1], [-1, 1, 0, 1],
     [-1, 1, 0, -1], [-1, -1, 0, 1], [-1, -1, 0, -1], [1, 1, 1, 0],
     [1, 1, -1, 0], [1, -1, 1, 0], [1, -1, -1, 0], [-1, 1, 1, 0],
     [-1, 1, -1, 0], [-1, -1, 1, 0], [-1, -1, -1, 0]],
    dtype=np.int32)

GRAD4 = ti.Vector.field(4, int, shape=len(grad4_np))
GRAD4.from_numpy(grad4_np)

perm_np = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
    154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19,
    98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
    251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235,
    249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176,
    115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29,
    24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91,
    90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26,
    197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56,
    87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27,
    166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92,
    41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
    209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
    164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202,
    38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17,
    182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
    221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110,
    79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242,
    193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239,
    107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50,
    45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
    141, 128, 195, 78, 66, 215, 61, 156, 180
],
                   dtype=np.int32)

PERM = ti.field(int, shape=512)
PERM.from_numpy(perm_np)

simplex_np = np.array(
    [[0, 1, 2, 3], [0, 1, 3, 2], [0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 0], [0, 2, 1, 3], [0, 0, 0, 0],
     [0, 3, 1, 2], [0, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
     [1, 3, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 3],
     [0, 0, 0, 0], [1, 3, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
     [2, 3, 0, 1], [2, 3, 1, 0], [1, 0, 2, 3], [1, 0, 3, 2], [0, 0, 0, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 3, 1], [0, 0, 0, 0], [2, 1, 3, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 1, 3], [0, 0, 0, 0],
     [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 1, 2], [3, 0, 2, 1], [0, 0, 0, 0],
     [3, 1, 2, 0], [2, 1, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
     [3, 1, 0, 2], [0, 0, 0, 0], [3, 2, 0, 1], [3, 2, 1, 0]],
    dtype=np.int32)

SIMPLEX = ti.Vector.field(4, int, shape=64)
SIMPLEX.from_numpy(simplex_np)


@ti.func
def noise4d(x, y, z, w):
    """4D simplex noise."""
    s = (x + y + z + w) * F4
    ijkl = tm.floor(tm.vec4(x, y, z, w) + s)
    t = (ijkl.x + ijkl.y + ijkl.z + ijkl.w) * G4
    xyzw0 = tm.vec4(x, y, z, w) - ijkl + t

    c = int(xyzw0.x > xyzw0.y) * 32 + int(xyzw0.x > xyzw0.z) * 16 + int(
        xyzw0.y > xyzw0.z) * 8 + int(xyzw0.x > xyzw0.w) * 4 + int(
            xyzw0.y > xyzw0.w) * 2 + int(xyzw0.z > xyzw0.w)
    ijkl1 = SIMPLEX[c] >= 3
    ijkl2 = SIMPLEX[c] >= 2
    ijkl3 = SIMPLEX[c] >= 1

    xyzw1 = xyzw0 - ijkl1 + G4
    xyzw2 = xyzw0 - ijkl2 + 2 * G4
    xyzw3 = xyzw0 - ijkl3 + 3 * G4
    xyzw4 = xyzw0 - 1 + 4 * G4

    I, J, K, L = ijkl.cast(int) & 255
    i1, j1, k1, l1 = ijkl1
    i2, j2, k2, l2 = ijkl2
    i3, j3, k3, l3 = ijkl3
    gi0 = PERM[I + PERM[J + PERM[K + PERM[L]]]] & 0x1f
    gi1 = PERM[I + i1 + PERM[J + j1 + PERM[K + k1 + PERM[L + l1]]]] & 0x1f
    gi2 = PERM[I + i2 + PERM[J + j2 + PERM[K + k2 + PERM[L + l2]]]] & 0x1f
    gi3 = PERM[I + i3 + PERM[J + j3 + PERM[K + k3 + PERM[L + l3]]]] & 0x1f
    gi4 = PERM[I + 1 + PERM[J + 1 + PERM[K + 1 + PERM[L + 1]]]] & 0x1f

    zs = [xyzw0, xyzw1, xyzw2, xyzw3, xyzw4]
    inds = [gi0, gi1, gi2, gi3, gi4]
    noise = 0.
    for i in ti.static(range(5)):
        t = 0.6 - tm.dot(zs[i], zs[i])
        if t >= 0:
            t *= t
            noise += t * t * tm.dot(zs[i], GRAD4[inds[i]])

    return 27.0 * noise


n = 40000
angs = np.random.uniform(0, 2 * np.pi, n)
rads = np.sqrt(np.random.uniform(0, 1, n))
initial_positions = np.zeros((n, 2), dtype=np.float32)
X, Y = initial_positions[:, 0], initial_positions[:, 1]
X[...] = rads * np.cos(angs)
Y[...] = rads * np.sin(angs)
intensity = np.power(1.001 - np.sqrt(X**2 + Y**2), 0.75)
pos = ti.Vector.field(2, float, shape=n)

scale = 0.6  # control sync between adjacent points
length = 0.65  # control amplitude of point wiggles


@ti.kernel
def update(
    time: float,  # pylint: disable=redefined-outer-name
    intensity: ti.types.ndarray(),  # pylint: disable=redefined-outer-name
    points: ti.types.ndarray(dtype=tm.vec2, ndim=1)):
    pos.fill(0)
    ct = 1.5 * ti.cos(2 * np.pi * time)
    st = 1.5 * ti.sin(2 * np.pi * time)
    for i in range(n):
        x, y = points[i]
        dx = noise4d(scale * x, scale * y, ct, st)
        dx *= intensity[i] * length
        dy = noise4d(100 + scale * x, 200 + scale * y, ct, st)
        dy *= intensity[i] * length
        pos[i] = x + dx, y + dy


size = 800
gui = ti.GUI("Alien Life", res=(size, size))
time = 0
theta = np.linspace(0, 2 * np.pi, 200)
begin = np.array([np.cos(theta)[:-1], np.sin(theta[:-1])]).transpose()
end = np.array([np.cos(theta)[1:], np.sin(theta[1:])]).transpose()
begin = (begin + 1) / 2
end = (end + 1) / 2

while gui.running:
    time += 1
    update(time * 0.002, intensity, initial_positions)
    gui.circles((pos.to_numpy() + 1) / 2, radius=1.5, color=0xFFFFFF)
    gui.lines(begin, end, radius=3, color=0xFFFFFF)
    gui.show()
