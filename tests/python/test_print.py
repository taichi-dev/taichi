import sys

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
@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(exclude=[ti.dx11, ti.amdgpu])
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        # make sure `%` doesn't break vprintf:
        print('hello, world! %s %d %f', 233, y)
        print('cool', x, 'well', y)

    func(666, 233.3)
    ti.sync()


@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
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


@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
def test_print_matrix_format():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f'hello {x[None]:.2f} world!')
        print(f'{(y[2] * k):.2e} {(x[None] / k):.8} {y[2]:.3}')

    func(233.3)
    ti.sync()


@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
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


@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
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


@test_utils.test(exclude=[ti.dx11, vk_on_mac, ti.amdgpu], debug=True)
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


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_python_scope_print_field():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.field(dtype=ti.f32, shape=3)

    print(x)
    print(y)
    print(z)


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
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


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_string_format_specifier_32():
    a = ti.field(ti.f32, 2)

    @ti.kernel
    def func(u: ti.u32, i: ti.i32):
        a[0] = ti.f32(1.111111)
        a[1] = ti.f32(2.222222)

        print("{:} {:}".format(a[0], a[1]))
        print("{:f} {:F}".format(a[0], a[1]))
        print("{:e} {:E}".format(a[0], a[1]))
        print("{:a} {:A}".format(a[0], a[1]))
        print("{:g} {:G}".format(a[0], a[1]))

        print("{:.2} {:.3}".format(a[0], a[1]))
        print("{:.2f} {:.3F}".format(a[0], a[1]))
        print("{:.2e} {:.3E}".format(a[0], a[1]))
        print("{:.2a} {:.3A}".format(a[0], a[1]))
        print("{:.2G} {:.3G}".format(a[0], a[1]))

        print("{a0:.2f} {a1:.3f}".format(a0=a[0], a1=a[1]))

        print("{:x}".format(u))
        print("{:X}".format(u))
        print("{:o}".format(u))
        print("{:u}".format(u))
        print("{:}".format(u))
        print("{name:x}".format(name=u))

        print("{:d}".format(i))
        print("{:i}".format(i))
        print("{:}".format(i))
        print("{name:}".format(name=i))

    func(0xdeadbeef, 2**31 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cpu, ti.cuda],
                 exclude=[ti.amdgpu, ti.vulkan],
                 debug=True)
def test_print_string_format_specifier_64():
    a = ti.field(ti.f64, 2)

    @ti.kernel
    def func(llu: ti.u64, lli: ti.i64):
        a[0] = ti.f64(1.111111111111)
        a[1] = ti.f64(2.222222222222)

        print("{:} {:}".format(a[0], a[1]))
        print("{:f} {:F}".format(a[0], a[1]))
        print("{:e} {:E}".format(a[0], a[1]))
        print("{:a} {:A}".format(a[0], a[1]))
        print("{:g} {:G}".format(a[0], a[1]))

        print("{:.2} {:.3}".format(a[0], a[1]))
        print("{:.2f} {:.3F}".format(a[0], a[1]))
        print("{:.2e} {:.3E}".format(a[0], a[1]))
        print("{:.2a} {:.3A}".format(a[0], a[1]))
        print("{:.2G} {:.3G}".format(a[0], a[1]))

        print("{a0:f} {a1:f}".format(a0=a[0], a1=a[1]))

        print("{:llx}".format(llu))
        print("{:llX}".format(llu))
        print("{:llo}".format(llu))
        print("{:llu}".format(llu))
        print("{:ll}".format(llu))
        print("{:x}".format(llu))
        print("{:X}".format(llu))
        print("{:o}".format(llu))
        print("{:u}".format(llu))
        print("{:}".format(llu))
        print("{name:llx}".format(name=llu))
        print("{name:x}".format(name=llu))

        print("{:lld}".format(lli))
        print("{:lli}".format(lli))
        print("{:ll}".format(lli))
        print("{:d}".format(lli))
        print("{:i}".format(lli))
        print("{:}".format(lli))
        print("{name:ll}".format(name=lli))

    func(0xcafebabedeadbeef, 2**63 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_string_format_specifier_vulkan_ul():
    @ti.kernel
    def func(llu: ti.u64):
        print("{:}".format(llu))
        print("{:u}".format(llu))
        print("{:lu}".format(llu))
        print("{name:lu}".format(name=llu))

        # FIXME: %lx works on vulkan bot %lX only prints lower 32 bits... why?
        print("{:lx}".format(llu))
        print("{name:lx}".format(name=llu))

        print("{:lX}".format(llu))
        print("{name:lX}".format(name=llu))

    func(0xcafebabedeadbeef)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_string_format_positional_arg():
    a = ti.field(ti.f32, 2)

    @ti.kernel
    def func(u: ti.u32):
        a[0] = ti.f32(1.111111)
        a[1] = ti.f32(2.222222)
        print("{0:.2f} {1:.3f}".format(a[0], a[1]))
        print("{1:.2f} {0:.3f}".format(a[1], a[0]))

        print("{a0:.2f} {0:.3f}".format(a[1], a0=a[0]))
        print("{0:.2f} {a1:.3f}".format(a[0], a1=a[1]))

        print("{a0:.2f} {:.3f}".format(a[1], a0=a[0]))
        print("{:.2f} {a1:.3f}".format(a[0], a1=a[1]))

        print("{0:x}".format(u))

        print(
            "a[0] = {0:.2f}, f = {name:.2f}, u = {u:x}, a[1] = {1:.3f}, a[1] = {1:.4f}"
            .format(a[0], a[1], name=42., u=u))

    func(0xdeadbeef)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_fstring():
    def foo1(x):
        return x + 1

    @ti.kernel
    def func(i: ti.i32, f: ti.f32):
        print(f'qwe {foo1(1)} {foo1(2) * 2 - 1} {i} {f} {4} {True} {1.23}')

    func(123, 4.56)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_fstring_specifier_32():
    a = ti.field(ti.f32, 2)

    @ti.kernel
    def func(u: ti.u32, i: ti.i32):
        a[0] = ti.f32(1.111111)
        a[1] = ti.f32(2.222222)

        print(f"{a[0]:} {a[1]:}")
        print(f"{a[0]:f} {a[1]:F}")
        print(f"{a[0]:e} {a[1]:E}")
        print(f"{a[0]:a} {a[1]:A}")
        print(f"{a[0]:g} {a[1]:G}")

        print(f"{a[0]:.2} {a[1]:.3}")
        print(f"{a[0]:.2f} {a[1]:.3F}")
        print(f"{a[0]:.2e} {a[1]:.3E}")
        print(f"{a[0]:.2a} {a[1]:.3A}")
        print(f"{a[0]:.2G} {a[1]:.3G}")

        print(f"{u:x}")
        print(f"{u:X}")
        print(f"{u:o}")
        print(f"{u:u}")
        print(f"{u:}")

        print(f"{i:d}")
        print(f"{i:i}")
        print(f"{i:}")

    func(0xdeadbeef, 2**31 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda],
                 exclude=[ti.vulkan, ti.amdgpu],
                 debug=True)
def test_print_fstring_specifier_64():
    a = ti.field(ti.f64, 2)

    @ti.kernel
    def func(u: ti.u64, i: ti.i64):
        a[0] = ti.f32(1.111111111111)
        a[1] = ti.f32(2.222222222222)

        print(f"{a[0]:} {a[1]:}")
        print(f"{a[0]:f} {a[1]:F}")
        print(f"{a[0]:e} {a[1]:E}")
        print(f"{a[0]:a} {a[1]:A}")
        print(f"{a[0]:g} {a[1]:G}")

        print(f"{a[0]:.2} {a[1]:.3}")
        print(f"{a[0]:.2f} {a[1]:.3F}")
        print(f"{a[0]:.2e} {a[1]:.3E}")
        print(f"{a[0]:.2a} {a[1]:.3A}")
        print(f"{a[0]:.2G} {a[1]:.3G}")

        print(f"{a[0]:.2f} {a[1]:.3f}")

        print(f"{u:x}")
        print(f"{u:X}")
        print(f"{u:o}")
        print(f"{u:u}")
        print(f"{u:}")

        print(f"{i:d}")
        print(f"{i:i}")
        print(f"{i:}")

    func(0xcafebabedeadbeef, 2**63 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[ti.amdgpu],
                 debug=True)
def test_print_fstring_specifier_vulkan_ul():
    @ti.kernel
    def func(llu: ti.u64):
        print(f"{llu:}")
        print(f"{llu:u}")
        print(f"{llu:lu}")

        # FIXME: %lx works on vulkan bot %lX only prints lower 32 bits... why?
        print(f"{llu:lx}")
        print(f"{llu:lX}")

    func(0xcafebabedeadbeef)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_u64():
    @ti.kernel
    def func(i: ti.u64):
        print("i =", i)

    func(2**64 - 1)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, ti.amdgpu],
                 debug=True)
def test_print_i64():
    @ti.kernel
    def func(i: ti.i64):
        print("i =", i)

    func(-2**63 + 2**31)
    ti.sync()


@test_utils.test(arch=[ti.cc, ti.cpu, ti.cuda, ti.vulkan],
                 exclude=[vk_on_mac, cuda_on_windows, ti.amdgpu],
                 debug=True)
def test_print_seq(capfd):
    @ti.kernel
    def foo():
        print("inside kernel")

    foo()
    print("outside kernel")
    out = capfd.readouterr().out
    assert "inside kernel\noutside kernel" in out
