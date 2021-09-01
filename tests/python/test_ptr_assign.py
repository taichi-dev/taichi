import taichi as ti


@ti.test()
def test_ptr_scalar():
    a = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def func(t: ti.f32):
        b = ti.static(a)
        c = ti.static(b)
        b[None] = b[None] * t
        c[None] = a[None] + t

    for x, y in zip(range(-5, 5), range(-4, 4)):
        a[None] = x
        func(y)
        assert a[None] == x * y + y


@ti.test()
def test_ptr_matrix():
    a = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())

    @ti.kernel
    def func(t: ti.f32):
        a[None] = [[2, 3], [4, 5]]
        b = ti.static(a)
        b[None][1, 0] = t

    for x in range(-5, 5):
        func(x)
        assert a[None][1, 0] == x


@ti.test()
def test_ptr_field():
    a = ti.field(dtype=ti.f32, shape=(3, 4))

    @ti.kernel
    def func(t: ti.f32):
        b = ti.static(a)
        b[1, 3] = b[1, 2] * t
        b[2, 0] = b[2, 1] + t

    for x, y in zip(range(-5, 5), range(-4, 4)):
        a[1, 2] = x
        a[2, 1] = x
        func(y)
        assert a[1, 3] == x * y
        assert a[2, 0] == x + y


@ti.test()
def test_pythonish_tuple_assign():
    a = ti.field(dtype=ti.f32, shape=())
    b = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def func(x: ti.f32, y: ti.f32):
        u, v = ti.static(b, a)
        u[None] = x
        v[None] = y

    for x, y in zip(range(-5, 5), range(-4, 4)):
        func(x, y)
        assert a[None] == y
        assert b[None] == x


@ti.test()
def test_ptr_func():
    a = ti.field(dtype=ti.f32, shape=())

    @ti.func
    def add2numbers(x, y):
        return x + y

    @ti.kernel
    def func():
        add = ti.static(add2numbers)
        a[None] = add(2, 3)

    func()
    assert a[None] == 5.0


@ti.test()
def test_ptr_class_func():
    @ti.data_oriented
    class MyClass:
        def __init__(self):
            self.a = ti.field(dtype=ti.f32, shape=())

        @ti.func
        def add2numbers(self, x, y):
            return x + y

        @ti.kernel
        def func(self):
            a, add = ti.static(self.a, self.add2numbers)
            a[None] = add(2, 3)

    obj = MyClass()
    obj.func()
    assert obj.a[None] == 5.0
