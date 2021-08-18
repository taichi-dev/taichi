import taichi as ti


def ti_support_atomic_more_than_add(test):
    return ti.archs_excluding(ti.vulkan)(
        test)  # TODO(changyu): support atomicSub/Mul/Min/Max on Vulkan backend


@ti.test()
def test_explicit_local_atomic_add():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            ti.atomic_add(a, i)
        A[None] = a

    func()
    assert A[None] == 45


@ti.test()
def test_implicit_local_atomic_add():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            a += i
        A[None] = a

    func()
    assert A[None] == 45


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_sub():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            ti.atomic_sub(a, i)
        A[None] = a

    func()
    assert A[None] == -45


@ti.test(exclude=ti.vulkan)
def test_implicit_local_atomic_sub():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            a -= i
        A[None] = a

    func()
    assert A[None] == -45


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_min():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 1000
        for i in range(10):
            ti.atomic_min(a, i)
        A[None] = a

    func()
    assert A[None] == 0


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_max():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = -1000
        for i in range(10):
            ti.atomic_max(a, i)
        A[None] = a

    func()
    assert A[None] == 9


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_and():
    A = ti.field(ti.i32, shape=())
    max_int = 2147483647

    @ti.kernel
    def func():
        a = 1023
        for i in range(10):
            ti.atomic_and(a, max_int - 2**i)
        A[None] = a

    func()
    assert A[None] == 0


@ti.test(exclude=ti.vulkan)
def test_implicit_local_atomic_and():
    A = ti.field(ti.i32, shape=())
    max_int = 2147483647

    @ti.kernel
    def func():
        a = 1023
        for i in range(10):
            a &= max_int - 2**i
        A[None] = a

    func()
    assert A[None] == 0


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_or():
    A = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            ti.atomic_or(a, 2**i)
        A[None] = a

    func()
    assert A[None] == 1023


@ti.test(exclude=ti.vulkan)
def test_implicit_local_atomic_or():
    A = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = 0
        for i in range(10):
            a |= 2**i
        A[None] = a

    func()
    assert A[None] == 1023


@ti.test(exclude=ti.vulkan)
def test_explicit_local_atomic_xor():
    A = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = 1023
        for i in range(10):
            ti.atomic_xor(a, 2**i)
        A[None] = a

    func()
    assert A[None] == 0


@ti.test(exclude=ti.vulkan)
def test_implicit_local_atomic_xor():
    A = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = 1023
        for i in range(10):
            a ^= 2**i
        A[None] = a

    func()
    assert A[None] == 0
