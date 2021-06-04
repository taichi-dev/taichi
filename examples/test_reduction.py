import time

import numpy as np

import taichi as ti

ti.init(ti.gpu)
n = 512 * 2
a = np.random.randn(n * n).reshape([n, n]).astype(np.float32)
b = ti.field(ti.f32, [n, n])
b.from_numpy(a)


@ti.kernel
def sum1() -> ti.f32:
    result = 0.0
    for i, j in b:
        #ti.atomic_add(result,b[i,j])
        result += b[i, j]
    return result


@ti.kernel
def sum2(range_n: ti.i32):
    for i in range(range_n):
        i1 = i // n
        j1 = i - i1
        i2 = (range_n + i) // n
        j2 = (range_n + i) - i2
        b[i1, j1] += b[i2, j2]
        #b[i1,j1]=max(b[i1,j1],b[i2,j2])


t1 = time.time()
print(a.sum())  #print(a.max())
t2 = time.time()
print(sum1())
t3 = time.time()
k = n**2 // 2
while k != 0:
    sum2(k)
    k //= 2
print(b[0])
t4 = time.time()
print('time numpy:', t4 - t3, 'time ti atomi_cadd:', t3 - t2,
      'time reduction:', t2 - t1)
input('enter to continue')
