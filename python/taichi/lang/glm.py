import taichi as ti

@ti.func
def fract(x):
    x - ti.floor(x)

@ti.func
def clamp(x, a, b):
    return ti.min(ti.max(x, a), b)

@ti.func
def mix(x, y, a):
    return x * (1 - a) + y * a

@ti.func
def step(e, x):
    ret = x
    if x < e:
        ret = 0
    else:
        ret = 1
    return ret

@ti.func
def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

@ti.func
def sign(x):
    ret = x
    if x > 0:
        ret = 1
    elif x == 0:
        ret = 0
    else:
        ret = -1
    return ret

@ti.func
def exp2(x):
    return ti.exp(x * ti.ln2)

@ti.func
def log2(x):
    return ti.log(x) * ti.log2e

@ti.func
def vec2(x, y):
    return ti.Vector([x, y])

@ti.func
def vec3(x, y, z):
    return ti.Vector([x, y, z])

@ti.func
def vec4(x, y, z, w):
    return ti.Vector([x, y, z, w])

@ti.func
def length(x):
    return x.norm()

@ti.func
def distance(x, y):
    return (x - y).norm()

@ti.func
def dot(x, y):
    return x.dot(y)

@ti.func
def cross(x, y):
    return x.cross(y)

@ti.func
def normalize(x):
    return ti.Vector.normalized(x)

@ti.func
def faceforward(n, i, r):
    ret = n
    if r.dot(i) >= 0:
        ret = -n
    return ret

@ti.func
def reflect(i, n):
    return i - 2 * n.dot(i) * n

@ti.func
def refract(i, n, eta):
    noi = n.dot(i)
    k = 1 - eta ** 2 * (1 - noi ** 2)
    if k < 0:
        for I in ti.static(ti.grouped(ti.ndrange(ret.m, ret.n))):
            ret[I] = 0
    else:
        ret = eta * i - (eta * noi + ti.sqrt(k)) * n
    return ret
