import pytest

import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_try():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        try:
            a = 0
        except:
            a = 1

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_for_else():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        for i in range(10):
            pass
        else:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_while_else():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        while True:
            pass
        else:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_loop_var_range():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in range(10):
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_loop_var_struct():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in x:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_loop_var_struct():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        j = 0
        for i, j in x:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_func_def_in_kernel():
    @ti.kernel
    def kernel():
        @ti.func
        def func():
            return 1

        print(func())

    kernel()


@ti.must_throw(ti.TaichiSyntaxError)
def test_func_def_in_func():
    @ti.func
    def func():
        @ti.func
        def func2():
            return 1

        return func2()

    @ti.kernel
    def kernel():
        print(func())

    kernel()


@ti.test(arch=ti.cpu)
def test_kernel_bad_argument_annotation():
    with pytest.raises(ti.KernelDefError, match='annotation'):

        @ti.kernel
        def kernel(x: 'bar'):
            print(x)


@ti.test(arch=ti.cpu)
def test_func_bad_argument_annotation():
    with pytest.raises(ti.KernelDefError, match='annotation'):

        @ti.func
        def func(x: 'foo'):
            print(x)


@ti.must_throw(ti.TaichiSyntaxError)
def test_nested_static():
    @ti.kernel
    def func():
        for i in ti.static(ti.static(range(1))):
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_nested_grouped():
    @ti.kernel
    def func():
        for i in ti.grouped(ti.grouped(range(1))):
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_nested_ndrange():
    @ti.kernel
    def func():
        for i in ti.ndrange(ti.ndrange(1)):
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_static_grouped_struct_for():
    val = ti.field(ti.i32)

    ti.root.dense(ti.ij, (1, 1)).place(val)

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(val)):
            pass

    test()


@ti.must_throw(ti.TaichiSyntaxError)
def test_is():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b is c

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_is_not():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b is not c

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_in():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b in c

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_not_in():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b not in c

    func()


@ti.test(arch=ti.cpu)
def test_func_multiple_return():
    @ti.func
    def safe_sqrt(a):
        if a > 0:
            return ti.sqrt(a)
        else:
            return 0.0

    @ti.kernel
    def kern(a: float):
        print(safe_sqrt(a))

    with pytest.raises(ti.TaichiSyntaxError,
                       match='cannot have multiple returns'):
        kern(-233)


@ti.test(arch=ti.cpu)
def test_func_multiple_return_in_static_if():
    @ti.func
    def safe_static_sqrt(a: ti.template()):
        if ti.static(a > 0):
            return ti.sqrt(a)
        else:
            return 0.0

    @ti.kernel
    def kern():
        print(safe_static_sqrt(-233))

    kern()


def test_func_def_inside_kernel():
    @ti.kernel
    def k():
        @ti.func
        def illegal():
            return 1

    with pytest.raises(ti.TaichiSyntaxError,
                       match='Function definition not allowed'):
        k()


def test_func_def_inside_func():
    @ti.func
    def f():
        @ti.func
        def illegal():
            return 1

    @ti.kernel
    def k():
        f()

    with pytest.raises(ti.TaichiSyntaxError,
                       match='Function definition not allowed'):
        k()
