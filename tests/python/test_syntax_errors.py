import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_try():
    x = ti.var(ti.f32)

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
    x = ti.var(ti.f32)

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
    x = ti.var(ti.f32)

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
    x = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in range(10):
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_loop_var_struct():
    x = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in x:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_loop_var_struct():
    x = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        j = 0
        for i, j in x:
            pass

    func()


@ti.must_throw(ti.TaichiSyntaxError)
def test_ternary():
    x = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        a = 0 if True else 123

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
    val = ti.var(ti.i32)

    ti.root.dense(ti.ij, (1, 1)).place(val)

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(val)):
            pass

    test()
