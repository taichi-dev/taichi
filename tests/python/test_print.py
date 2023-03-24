import pytest

import taichi as ti
from tests import test_utils

#TODO: validation layer support on macos vulkan backend is not working.
vk_on_mac = (ti.vulkan, 'Darwin')

#TODO: capfd doesn't function well on CUDA backend on Windows
cuda_on_windows = (ti.cuda, 'Windows')


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support print() or 64-bit data
# While OpenGL does support print, but not 64-bit data
@pytest.mark.parametrize('dt', ti.types.primitive_types.all_types)
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
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        # make sure `%` doesn't break vprintf:
        print('hello, world! %s %d %f', 233, y)
        print('cool', x, 'well', y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_matrix():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print('hello', x[None], 'world!')
        print(y[2] * k, x[None] / k, y[2])

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_matrix_string_format():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print('hello {} world!'.format(x[None]))
        print('{} {} {}'.format(y[2] * k, x[None] / k, y[2]))

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows],
                 debug=True)
def test_print_matrix_string_format_with_spec(capfd):
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print('hello {:.2f} world!'.format(x[None]))
        print('{:.3f} {:e} {:.2}'.format(y[2] * k, x[None] / k, y[2]))
        print('hello {:+d} world!'.format(z[None]))

    func(233.3)
    ti.sync()

    out, err = capfd.readouterr()
    # TODO: format specifiers are ignored for now
    expected_out = '''hello [[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]] world!
[233.300003, 233.300003, 233.300003] [[-0.004286, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]] [1.000000, 1.000000, 1.000000]
hello [[0, 0, 0], [0, 0, 0]] world!
'''
    assert out == expected_out and err == ''


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_matrix_string_format_with_spec_mismatch():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def test_x():
        print('hello {:.2d} world!'.format(x[None]))

    @ti.kernel
    def test_y(k: ti.f32):
        print('{:- #0.233lli} {:e} {:.2}'.format(y[2] * k, x[None] / k, y[2]))

    @ti.kernel
    def test_z():
        print('hello {:.2e} world!'.format(z[None]))

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'.2e' doesn't match 'i32'."):
        test_z()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_matrix_fstring():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f'hello {x[None]} world!')
        print(f'{y[2] * k} {x[None] / k} {y[2]}')

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows],
                 debug=True)
def test_print_matrix_fstring_with_spec(capfd):
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f'hello {x[None]:.2f} world!')
        print(f'{(y[2] * k):.3f} {(x[None] / k):e} {y[2]:.2}')
        print(f'hello {z[None]:+d} world!')

    func(233.3)
    ti.sync()

    out, err = capfd.readouterr()
    # TODO: format specifiers are ignored for now
    expected_out = '''hello [[-1.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]] world!
[233.300003, 233.300003, 233.300003] [[-0.004286, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]] [1.000000, 1.000000, 1.000000]
hello [[0, 0, 0], [0, 0, 0]] world!
'''
    assert out == expected_out and err == ''


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_matrix_fstring_with_spec_mismatch():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.Matrix.field(2, 3, dtype=ti.i32, shape=())

    @ti.kernel
    def test_x():
        print(f'hello {x[None]:.2d} world!')

    @ti.kernel
    def test_y(k: ti.f32):
        print(f'{(y[2] * k):- #0.233lli} {(x[None] / k):e} {y[2]:.2}')

    @ti.kernel
    def test_z():
        print(f'hello {z[None]:.2e} world!')

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(ti.TaichiTypeError,
                       match=r"'.2e' doesn't match 'i32'."):
        test_z()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_sep_end():
    @ti.kernel
    def func():
        # hello 42 world!
        print('hello', 42, 'world!')
        # hello 42 Taichi 233 world!
        print('hello', 42, 'Tai', end='')
        print('chi', 233, 'world!')
        # hello42world!
        print('hello', 42, 'world!', sep='')
        # '  ' (with no newline)
        print('  ', end='')
        # 'helloaswd42qwer'
        print('  ', 42, sep='aswd', end='qwer')

    func()
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_multiple_threads():
    x = ti.field(dtype=ti.f32, shape=(128, ))

    @ti.kernel
    def func(k: ti.f32):
        for i in x:
            x[i] = i * k
            print('x[', i, ']=', x[i])

    func(0.1)
    ti.sync()
    func(10.0)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_list():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(2, 3))
    y = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        w = [k, x.shape]
        print(w + [y.n])  # [233.3, [2, 3], 3]
        print(x.shape)  # [2, 3]
        print(y.shape)  # []
        z = (1, )
        print([1, k**2, k + 1])  # [1, 233.3, 234.3]
        print(z)  # [1]
        print([y[None], z])  # [[0, 0, 0], [1]]
        print([])  # []

    func(233.3)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_python_scope_print_field():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.field(dtype=ti.f32, shape=3)

    print(x)
    print(y)
    print(z)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows],
                 debug=True)
def test_print_string_format_with_spec(capfd):
    @ti.kernel
    def func(k: ti.f32):
        print(123)
        print("{:d} abc".format(123))
        print("{: } {:+} {:10d}".format(1, 2, 3))
        print("{:.2} {name:D} {value:d}".format(k, name=999, value=123))
        name = 123.4
        value = 456.7
        print("{:.2e} {name:.3G} {value:.4f}".format(k, name=name,
                                                     value=value))

    func(233.3)
    ti.sync()
    out, err = capfd.readouterr()
    # TODO: format specifiers are ignored for now
    expected_out = '''123
123 abc
1 2 3
233.300003 999 123
233.300003 123.400002 456.700012
'''
    assert out == expected_out and err == ''


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_string_format_with_spec_mismatch():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def test_i(i: ti.i32):
        print('{:u}'.format(foo1(i)))

    @ti.kernel
    def test_u(u: ti.u32):
        print('{:d}'.format(foo1(u)))

    @ti.kernel
    def test_f(u: ti.f32):
        print('{:i}'.format(foo1(u)))

    with pytest.raises(ti.TaichiTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_fstring():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def func(i: ti.i32, f: ti.f32):
        print(f'qwe {foo1(1)} {foo1(2) * 2 - 1} {i} {f} {4} {True} {1.23}')

    func(123, 4.56)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows],
                 debug=True)
def test_print_fstring_with_spec(capfd):
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def func(i: ti.i32, f: ti.f32):
        print(
            f'qwe {foo1(1):d} {(foo1(2) * 2 - 1):10d} {i} {f:.1f} {4} {True} {1.23}'
        )

    func(123, 4.56)
    ti.sync()
    out, err = capfd.readouterr()
    # TODO: format specifiers are ignored for now
    expected_out = '''qwe 2 5 123 4.560000 4 True 1.23
'''
    assert out == expected_out and err == ''


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_fstring_with_spec_mismatch():
    @ti.func
    def foo1(x):
        return x + 1

    @ti.kernel
    def test_i(i: ti.i32):
        print(f'{foo1(i):u}')

    @ti.kernel
    def test_u(u: ti.u32):
        print(f'{foo1(u):d}')

    @ti.kernel
    def test_f(u: ti.f32):
        print(f'{foo1(u):i}')

    with pytest.raises(ti.TaichiTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(ti.TaichiTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_u64():
    @ti.kernel
    def func(i: ti.u64):
        print("i =", i)

    func(2**64 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac],
                 debug=True)
def test_print_i64():
    @ti.kernel
    def func(i: ti.i64):
        print("i =", i)

    func(-2**63 + 2**31)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows],
                 debug=True)
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
