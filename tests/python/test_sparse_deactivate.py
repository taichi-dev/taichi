import taichi as ti


@ti.test(require=ti.extension.sparse)
def test_pointer():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32, shape=())

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)

    s[None] = 0

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[19] = 1
    func()
    assert s[None] == 32

    @ti.kernel
    def deactivate():
        ti.deactivate(ptr, 0)

    deactivate()
    s[None] = 0
    func()
    assert s[None] == 16


@ti.test(require=ti.extension.sparse)
def test_pointer1():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)
    ti.root.place(s)

    s[None] = 0

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[19] = 1
    x[20] = 1
    x[45] = 1
    func()
    assert s[None] == 48

    @ti.kernel
    def deactivate():
        ti.deactivate(ptr, 0)

    deactivate()
    s[None] = 0
    func()
    assert s[None] == 32


@ti.test(require=ti.extension.sparse)
def test_pointer2():
    x = ti.field(ti.f32)

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)

    @ti.kernel
    def func():
        for i in range(n * n):
            x[i] = 1.0

    @ti.kernel
    def set10():
        x[10] = 10.0

    @ti.kernel
    def clear():
        for i in ptr:
            ti.deactivate(ptr, i)

    func()
    clear()

    for i in range(n * n):
        assert x[i] == 0.0

    set10()

    for i in range(n * n):
        if i != 10:
            assert x[i] == 0.0
        else:
            assert x[i] == 10.0


@ti.test(require=ti.extension.sparse)
def test_pointer3():
    x = ti.field(ti.f32)
    x_temp = ti.field(ti.f32)

    n = 16

    ptr1 = ti.root.pointer(ti.ij, n)
    ptr1.dense(ti.ij, n).place(x)
    ptr2 = ti.root.pointer(ti.ij, n)
    ptr2.dense(ti.ij, n).place(x_temp)

    @ti.kernel
    def fill():
        for j in range(n * n):
            for i in range(n * n):
                x[i, j] = i + j

    @ti.kernel
    def fill2():
        for i, j in x_temp:
            if x_temp[i, j] < 100:
                x[i, j] = x_temp[i, j]

    @ti.kernel
    def copy_to_temp():
        for i, j in x:
            x_temp[i, j] = x[i, j]

    @ti.kernel
    def copy_from_temp():
        for i, j in x_temp:
            x[i, j] = x_temp[i, j]

    @ti.kernel
    def clear():
        for i, j in ptr1:
            ti.deactivate(ptr1, [i, j])

    @ti.kernel
    def clear_temp():
        for i, j in ptr2:
            ti.deactivate(ptr2, [i, j])

    fill()
    copy_to_temp()
    clear()
    fill2()
    clear_temp()

    for itr in range(100):
        copy_to_temp()
        clear()
        copy_from_temp()
        clear_temp()

        xn = x.to_numpy()
        for j in range(n * n):
            for i in range(n * n):
                if i + j < 100:
                    assert xn[i, j] == i + j


@ti.test(require=ti.extension.sparse)
def test_dynamic():
    x = ti.field(ti.i32)
    s = ti.field(ti.i32)

    n = 16

    lst = ti.root.dense(ti.i, n).dynamic(ti.j, 4096)
    lst.place(x)
    ti.root.dense(ti.i, n).place(s)

    @ti.kernel
    def func(mul: ti.i32):
        for i in range(n):
            for j in range(i * i * mul):
                ti.append(lst, i, j)

    @ti.kernel
    def fetch_length():
        for i in range(n):
            s[i] = ti.length(lst, i)

    func(1)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i

    @ti.kernel
    def clear():
        for i in range(n):
            ti.deactivate(lst, [i])

    func(2)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i * 3

    clear()
    fetch_length()
    for i in range(n):
        assert s[i] == 0

    func(4)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i * 4
