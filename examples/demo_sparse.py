import taichi as ti

ti.init(arch=ti.cpu, print_ir=True)

n = 512
x = ti.var(ti.f32)
res = n + n // 4 + n // 16 + n // 64
img = ti.var(ti.f32, shape=(res, res))

block1 = ti.root.pointer(ti.ij, n // 64)
block2 = block1.pointer(ti.ij, 4)
block3 = block2.pointer(ti.ij, 4)
block3.dense(ti.ij, 4).place(x)


@ti.kernel
def activate():
    for i, j in ti.ndrange(n, n):
        if i < j:
            x[i, j] = 1


@ti.func
def scatter(i):
    return i + i // 4 + i // 16 + i // 64 + 2


@ti.kernel
def paint():
    for i, j in ti.ndrange(n, n):
        t = x[i, j]
        t += ti.is_active(block1, [i, j])
        t += ti.is_active(block2, [i, j])
        t += ti.is_active(block3, [i, j])
        img[scatter(i), scatter(j)] = t / 4


@ti.kernel
def deact():
    for i, j in block2:
        print(i, j)
        if i % 32 == 0:
            ti.deactivate(block2, [i, j])
    # ti.deactivate(block2, [100, 100])
    # ti.deactivate(block1, [100, 100])


img.fill(0.05)

gui = ti.GUI('Sparse Grids', (res, res))

for i in range(100000):
    block1.deactivate_all()
    activate()
    deact()
    paint()
    gui.set_image(img.to_numpy())
    gui.show()
