import taichi as ti


@ti.all_archs
def test_fibonacci():
    @ti.kernel
    def ti_fibonacci(n: ti.i32) -> ti.i32:
        a, b = 0, 1
        # This is to make the inner for loop serial on purpose...
        for _ in range(1):
            for i in range(n):
                a, b = b, a + b
        return b

    def py_fibonacci(n):
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return b

    for n in range(5):
        assert ti_fibonacci(n) == py_fibonacci(n)


@ti.host_arch_only
def test_assign2():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None] = 2, 3

    func()
    assert a[None] == 2
    assert b[None] == 3


@ti.host_arch_only
@ti.must_throw(ValueError)
def test_assign2_mismatch3():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None] = 2, 3, 4

    func()


@ti.host_arch_only
@ti.must_throw(TypeError)
def test_assign2_mismatch1():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None] = 2

    func()


@ti.host_arch_only
def test_swap2():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None] = b[None], a[None]

    a[None] = 2
    b[None] = 3
    func()
    assert a[None] == 3
    assert b[None] == 2


@ti.host_arch_only
def test_assign2_static():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        # XXX: why a, b = ti.static(b, a) doesn't work?
        c, d = ti.static(b, a)
        c[None], d[None] = 2, 3

    func()
    assert a[None] == 3
    assert b[None] == 2


@ti.host_arch_only
def test_swap3():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())
    c = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a[None], b[None], c[None] = b[None], c[None], a[None]

    a[None] = 2
    b[None] = 3
    c[None] = 4
    func()
    assert a[None] == 3
    assert b[None] == 4
    assert c[None] == 2


@ti.host_arch_only
def test_unpack_from_tuple():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())
    c = ti.field(ti.f32, ())

    list = [2, 3, 4]

    @ti.kernel
    def func():
        a[None], b[None], c[None] = list

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4


@ti.host_arch_only
@ti.must_throw(ValueError)
def test_unpack_mismatch_tuple():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    list = [2, 3, 4]

    @ti.kernel
    def func():
        a[None], b[None] = list

    func()


@ti.host_arch_only
def test_unpack_from_vector():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())
    c = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        vector = ti.Vector([2, 3, 4])
        a[None], b[None], c[None] = vector

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4


@ti.host_arch_only
@ti.must_throw(ValueError)
def test_unpack_mismatch_vector():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        vector = ti.Vector([2, 3, 4])
        a[None], b[None] = vector

    func()


@ti.host_arch_only
@ti.must_throw(TypeError)
def test_unpack_mismatch_type():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())

    bad = 12

    @ti.kernel
    def func():
        a[None], b[None] = bad

    func()


@ti.host_arch_only
@ti.must_throw(ValueError)
def test_unpack_mismatch_matrix():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())
    c = ti.field(ti.f32, ())
    d = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        bad = ti.Matrix([[2, 3], [4, 5]])
        a[None], b[None], c[None], d[None] = bad

    func()


@ti.host_arch_only
def test_unpack_from_shape():
    a = ti.field(ti.f32, ())
    b = ti.field(ti.f32, ())
    c = ti.field(ti.f32, ())
    d = ti.field(ti.f32, (2, 3, 4))

    @ti.kernel
    def func():
        a[None], b[None], c[None] = d.shape

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4
