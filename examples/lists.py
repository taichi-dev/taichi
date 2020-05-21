import taichi as ti

x = ti.var(ti.i32)
l = ti.var(ti.i32)
n = 16

ti.init()  #print_preprocessed=True)


@ti.layout
def lists():
    ti.root.dense(ti.i, n).dynamic(ti.j, n).place(x)
    ti.root.dense(ti.i, n).place(l)


@ti.kernel
def make_lists():
    for i in range(n):
        for j in range(i):
            ti.append(x.parent(), i, j * j)
        l[i] = ti.length(x.parent(), i)


make_lists()

for i in range(n):
    assert l[i] == i
    for j in range(n):
        assert x[i, j] == (j * j if j < i else 0)
