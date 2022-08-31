import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_try():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        try:
            a = 0
        except:
            a = 1

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_for_else():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        for i in range(10):
            pass
        else:
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_while_else():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        while True:
            pass
        else:
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_raise():
    @ti.kernel
    def foo():
        raise Exception()

    with pytest.raises(ti.TaichiSyntaxError,
                       match='Unsupported node "Raise"') as e:
        foo()


@test_utils.test()
def test_loop_var_range():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in range(10):
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_loop_var_struct():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        i = 0
        for i in x:
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_loop_var_struct():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        j = 0
        for i, j in x:
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_func_def_in_kernel():
    @ti.kernel
    def kernel():
        @ti.func
        def func():
            return 1

        print(func())

    with pytest.raises(ti.TaichiCompilationError):
        kernel()


@test_utils.test()
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

    with pytest.raises(ti.TaichiCompilationError):
        kernel()


@test_utils.test(arch=ti.cpu)
def test_kernel_bad_argument_annotation():
    with pytest.raises(ti.TaichiSyntaxError, match='annotation'):

        @ti.kernel
        def kernel(x: 'bar'):
            print(x)


@test_utils.test(arch=ti.cpu)
def test_func_bad_argument_annotation():
    with pytest.raises(ti.TaichiSyntaxError, match='annotation'):

        @ti.func
        def func(x: 'foo'):
            print(x)


@test_utils.test()
def test_nested_static():
    @ti.kernel
    def func():
        for i in ti.static(ti.static(range(1))):
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_nested_grouped():
    @ti.kernel
    def func():
        for i in ti.grouped(ti.grouped(range(1))):
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_nested_ndrange():
    @ti.kernel
    def func():
        for i in ti.ndrange(ti.ndrange(1)):
            pass

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_static_grouped_struct_for():
    val = ti.field(ti.i32)

    ti.root.dense(ti.ij, (1, 1)).place(val)

    @ti.kernel
    def test():
        for I in ti.static(ti.grouped(val)):
            pass

    with pytest.raises(ti.TaichiCompilationError):
        test()


@test_utils.test()
def test_is():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b is c

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_is_not():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b is not c

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_in():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b in c

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_not_in():
    b = ti.field(ti.i32, shape=())
    c = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        a = b not in c

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_expr_set():
    @ti.kernel
    def func():
        x = {2, 4, 6}

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test()
def test_redefining_template_args():
    @ti.kernel
    def foo(a: ti.template()):
        a = 5

    with pytest.raises(
            ti.TaichiSyntaxError,
            match=
            "Variable 'a' cannot be assigned. Maybe it is not a Taichi object?"
    ):
        foo(1)


@test_utils.test()
def test_break_in_outermost_for():
    @ti.kernel
    def foo():
        for i in range(10):
            break

    with pytest.raises(ti.TaichiSyntaxError,
                       match="Cannot break in the outermost loop"):
        foo()


@test_utils.test()
def test_funcdef_in_kernel():
    @ti.kernel
    def foo():
        def bar():
            pass

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="Function definition is not allowed in 'ti.kernel'"):
        foo()


@test_utils.test()
def test_funcdef_in_func():
    @ti.func
    def foo():
        def bar():
            pass

    @ti.kernel
    def baz():
        foo()

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="Function definition is not allowed in 'ti.func'"):
        baz()


@test_utils.test()
def test_continue_in_static_for_in_non_static_if():
    @ti.kernel
    def test_static_loop():
        for i in ti.static(range(5)):
            x = 0.1
            if x == 0.0:
                continue

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="You are trying to `continue` a static `for` loop"):
        test_static_loop()


@test_utils.test()
def test_break_in_static_for_in_non_static_if():
    @ti.kernel
    def test_static_loop():
        for i in ti.static(range(5)):
            x = 0.1
            if x == 0.0:
                break

    with pytest.raises(ti.TaichiSyntaxError,
                       match="You are trying to `break` a static `for` loop"):
        test_static_loop()
