import sys

import pytest

import taichi as ti
from tests import test_utils

# TODO: validation layer support on macos vulkan backend is not working.
vk_on_mac = (ti.vulkan, "Darwin")

# TODO: capfd doesn't function well on CUDA backend on Windows
cuda_on_windows = (ti.cuda, "Windows")

if sys.version_info >= (3, 8):
    # Import the test case only if the Python version is >= 3.8
    from .py38_only import (
        test_print_docs_matrix_self_documenting_exp,
        test_print_docs_scalar_self_documenting_exp,
    )


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support print() or 64-bit data
# While OpenGL does support print, but not 64-bit data
@pytest.mark.parametrize("dt", ti.types.primitive_types.all_types)
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_print(dt):
    @ti.kernel
    def func():
        print(ti.cast(123.4, dt))

    func()
    # Discussion: https://github.com/taichi-dev/taichi/issues/1063#issuecomment-636421904
    # Synchronize to prevent cross-test failure of print:
    ti.sync()


# TODO: As described by @k-ye above, what we want to ensure
#       is that, the content shows on console is *correct*.
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        # make sure `%` doesn't break vprintf:
        print("hello, world! %s %d %f", 233, y)
        print("cool", x, "well", y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("hello", x[None], "world!")
        print(y[2] * k, x[None] / k, y[2])

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_string_format():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("hello {} world!".format(x[None]))
        print("{} {} {}".format(y[2] * k, x[None] / k, y[2]))

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_matrix_string_format_with_spec(capfd):
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("hello {:.2f} world!".format(x[None]))
        print("{:.3f} {:e} {:.2}".format(y[2] * k, x[None] / k, y[2]))
        print("hello {:.10d} world!".format(z[None]))

    func(233.3)
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """hello [[-1.00, 0.00, 0.00], [0.00, 0.00, 0.00]] world!
[233.300, 233.300, 233.300] [[-4.286326e-03, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]] [1.00, 1.00, 1.00]
hello [[0000000000, 0000000000, 0000000000], [0000000000, 0000000000, 0000000000]] world!
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_string_format_with_spec_mismatch():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def test_x():
        print("hello {:.2d} world!".format(x[None]))

    @ti.kernel
    def test_y(k: ti.f32):
        print("{:- #0.233lli} {:e} {:.2}".format(y[2] * k, x[None] / k, y[2]))

    @ti.kernel
    def test_z():
        print("hello {:.2e} world!".format(z[None]))

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(ti.TaichiTypeError, match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(ti.TaichiTypeError, match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(ti.TaichiTypeError, match=r"'.2e' doesn't match 'i32'."):
        test_z()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_fstring():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f"hello {x[None]} world!")
        print(f"{y[2] * k} {x[None] / k} {y[2]}")

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_matrix_fstring_with_spec(capfd):
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f"hello {x[None]:.2f} world!")
        print(f"{(y[2] * k):.3f} {(x[None] / k):e} {y[2]:.2}")
        print(f"hello {z[None]:.2d} world!")

    func(233.3)
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """hello [[-1.00, 0.00, 0.00], [0.00, 0.00, 0.00]] world!
[233.300, 233.300, 233.300] [[-4.286326e-03, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]] [1.00, 1.00, 1.00]
hello [[00, 00, 00], [00, 00, 00]] world!
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_fstring_with_spec_mismatch():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def test_x():
        print(f"hello {x[None]:.2d} world!")

    @ti.kernel
    def test_y(k: ti.f32):
        print(f"{(y[2] * k):- #0.233lli} {(x[None] / k):e} {y[2]:.2}")

    @ti.kernel
    def test_z():
        print(f"hello {z[None]:.2e} world!")

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(ti.TaichiTypeError, match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(ti.TaichiTypeError, match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(ti.TaichiTypeError, match=r"'.2e' doesn't match 'i32'."):
        test_z()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_scalar(capfd):
    a = ti.field(ti.f32, 4)

    @ti.kernel
    def func():
        a[0] = 1.0

        # comma-separated string
        print("a[0] =", a[0])

        # f-string
        print(f"a[0] = {a[0]}")
        # with format specifier
        print(f"a[0] = {a[0]:.1f}")
        # without conversion
        print(f"a[0] = {a[0]:.1}")

        # formatted string via `str.format()` method
        print("a[0] = {}".format(a[0]))
        # with format specifier
        print("a[0] = {:.1f}".format(a[0]))
        # without conversion
        print("a[0] = {:.1}".format(a[0]))
        # with positional arguments
        print("a[3] = {3:.3f}, a[2] = {2:.2f}, a[1] = {1:.1f}, a[0] = {0:.0f}".format(a[0], a[1], a[2], a[3]))

    func()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """a[0] = 1.000000
a[0] = 1.000000
a[0] = 1.0
a[0] = 1.0
a[0] = 1.000000
a[0] = 1.0
a[0] = 1.0
a[3] = 0.000, a[2] = 0.00, a[1] = 0.0, a[0] = 1
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_matrix(capfd):
    a = ti.field(ti.f32, 4)

    @ti.kernel
    def func():
        m = ti.Matrix([[2e1, 3e2, 4e3], [5e4, 6e5, 7e6]], ti.f32)

        # comma-seperated string is supported
        print("m =", m)

        # f-string is supported
        print(f"m = {m}")
        # can with format specifier
        print(f"m = {m:.1f}")
        # can omitting conversion
        print(f"m = {m:.1}")

        # formatted string via `str.format()` method is supported
        print("m = {}".format(m))
        # can with format specifier
        print("m = {:e}".format(m))
        # and can omitting conversion
        print("m = {:.1}".format(m))

    func()
    ti.sync()

    out, err = capfd.readouterr()
    expected_out = """m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]
m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]
m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
m = [[2.000000e+01, 3.000000e+02, 4.000000e+03], [5.000000e+04, 6.000000e+05, 7.000000e+06]]
m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_sep_end():
    @ti.kernel
    def func():
        # hello 42 world!
        print("hello", 42, "world!")
        # hello 42 Taichi 233 world!
        print("hello", 42, "Tai", end="")
        print("chi", 233, "world!")
        # hello42world!
        print("hello", 42, "world!", sep="")
        # '  ' (with no newline)
        print("  ", end="")
        # 'helloaswd42qwer'
        print("  ", 42, sep="aswd", end="qwer")

    func()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_multiple_threads():
    x = ti.field(dtype=ti.f32, shape=(128,))

    @ti.kernel
    def func(k: ti.f32):
        for i in x:
            x[i] = i * k
            print("x[", i, "]=", x[i])

    func(0.1)
    ti.sync()
    func(10.0)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_list():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(2, 3))
    y = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        w = [k, x.shape]
        print(w + [y.n])  # [233.3, [2, 3], 3]
        print(x.shape)  # [2, 3]
        print(y.shape)  # []
        z = (1,)
        print([1, k**2, k + 1])  # [1, 233.3, 234.3]
        print(z)  # [1]
        print([y[None], z])  # [[0, 0, 0], [1]]
        print([])  # []

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_python_scope_print_field():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.field(dtype=ti.f32, shape=3)

    print(x)
    print(y)
    print(z)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format():
    @ti.kernel
    def func(k: ti.f32):
        print(123)
        print("{} abc".format(123))
        print("{} {} {}".format(1, 2, 3))
        print("{} {name} {value}".format(k, name=999, value=123))
        name = 123.4
        value = 456.7
        print("{} {name} {value}".format(k, name=name, value=value))

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_spec(capfd):
    @ti.kernel
    def func(k: ti.f32):
        print(123)
        print("{:d} abc".format(123))
        print("{:i} {:.1} {:.10d}".format(1, 2, 3))
        print("{:.2} {name:i} {value:d}".format(k, name=999, value=123))
        name = 123.4
        value = 456.7
        print("{:.2e} {name:.3G} {value:.4f}".format(k, name=name, value=value))

    func(233.3)
    ti.sync()
    out, err = capfd.readouterr()
    expected_out = """123
123 abc
1 2 0000000003
233.30 999 123
2.33e+02 123 456.7000
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format_with_spec_mismatch():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def test_i(i: ti.i32):
        print("{:u}".format(foo1(i)))

    @ti.kernel
    def test_u(u: ti.u32):
        print("{:d}".format(foo1(u)))

    @ti.kernel
    def test_f(u: ti.f32):
        print("{:i}".format(foo1(u)))

    with pytest.raises(ti.TaichiTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_positional_arg(capfd):
    @ti.kernel
    def func(k: ti.f32):
        print("{0} {1} {2}".format(1, 2, 3))
        print("{2} {1} {}".format(3, 2, 1))
        print("{2} {} {1} {k} {0} {k} {0} {k}".format(3, 2, 1, k=k))

    func(233.3)
    ti.sync()
    out, err = capfd.readouterr()
    expected_out = """1 2 3
1 2 3
1 3 2 233.300003 3 233.300003 3 233.300003
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_positional_arg_with_spec(capfd):
    @ti.kernel
    def func(k: ti.f32):
        print("{0:d} {1:} {2:i}".format(1, 2, 3))
        print("{2:d} {1:.2} {:.10}".format(3, 2, 1))
        print("{2:.1} {:.2} {1:.3} {k:.4e} {0:.5} {k:.5f} {0:.5} {k:.4g}".format(3.0, 2.0, 1.0, k=k))

    func(233.3)
    ti.sync()
    out, err = capfd.readouterr()
    expected_out = """1 2 3
1 02 0000000003
1.0 3.00 2.000 2.3330e+02 3.00000 233.30000 3.00000 233.3
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format_with_positional_arg_mismatch():
    @ti.kernel
    def func(k: ti.f32):
        print("{0} {1} {2}".format(1, 2))
        print("{2} {1} {}".format(3, 2, 1))
        print("{0} {} {0} {k} {0} {k}".format(1, k=k))

    @ti.kernel
    def func_k_not_used(k: ti.f32):
        print("".format(k=k))

    @ti.kernel
    def func_k_not_defined():
        print("{k}".format())

    @ti.kernel
    def func_more_args():
        print("{0} {1} {2}".format(1, 2, 3, 4))

    @ti.kernel
    def func_less_args():
        print("{0} {1} {2}".format(1, 2))

    with pytest.raises(
        ti.TaichiSyntaxError,
        match=r"Expected 3 positional argument\(s\), but received 4 instead.",
    ):
        func_more_args()
    with pytest.raises(
        ti.TaichiSyntaxError,
        match=r"Expected 3 positional argument\(s\), but received 2 instead.",
    ):
        func_less_args()
    with pytest.raises(ti.TaichiSyntaxError, match=r"Keyword 'k' not used."):
        func_k_not_used(233.3)
    with pytest.raises(ti.TaichiSyntaxError, match=r"Keyword 'k' not found."):
        func_k_not_defined()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_fstring():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def func(i: ti.i32, f: ti.f32):
        print(f"qwe {foo1(1)} {foo1(2) * 2 - 1} {i} {f} {4} {True} {1.23}")

    func(123, 4.56)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_fstring_with_spec(capfd):
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def func(i: ti.i32, f: ti.f32):
        print(f"qwe {foo1(1):d} {(foo1(2) * 2 - 1):.10d} {i} {f:.1f} {4} {True} {1.23}")

    func(123, 4.56)
    ti.sync()
    out, err = capfd.readouterr()
    expected_out = """qwe 2 0000000005 123 4.6 4 True 1.23
"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_fstring_with_spec_mismatch():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def test_i(i: ti.i32):
        print(f"{foo1(i):u}")

    @ti.kernel
    def test_u(u: ti.u32):
        print(f"{foo1(u):d}")

    @ti.kernel
    def test_f(u: ti.f32):
        print(f"{foo1(u):i}")

    with pytest.raises(ti.TaichiTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_u64():
    @ti.kernel
    def func(i: ti.u64):
        print("i =", i)

    func(2**64 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_i64():
    @ti.kernel
    def func(i: ti.i64):
        print("i =", i)

    func(-(2**63) + 2**31)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_seq(capfd):
    @ti.kernel
    def foo():
        print("inside kernel")

    foo()
    print("outside kernel")
    out = capfd.readouterr().out
    assert "inside kernel\noutside kernel" in out


@test_utils.test(arch=[ti.cpu, ti.cuda], print_ir=True, debug=True)
def test_fp16_print_ir():
    half2 = ti.types.vector(n=2, dtype=ti.f16)

    @ti.kernel
    def test():
        x = half2(1.0)
        y = half2(2.0)

        for i in range(2):
            x[i] = y[i]
            print(x[i])

    test()
