import taichi as ti
from taichi import make_temp_file
import sys, os


def test_without_init():
    # We want to check if Taichi works well without ``ti.init()``.
    # But in test ``ti.init()`` will always be called in last ``@ti.all_archs``.
    # So we have to create a new Taichi instance, i.e. test in a sandbox.
    content = '''
import taichi as ti
assert ti.cfg.arch == ti.cpu

x = ti.var(ti.i32, (2, 3))
assert x.shape == (2, 3)

x[1, 2] = 4
assert x[1, 2] == 4
'''
    filename = make_temp_file()
    with open(filename, 'w') as f:
        f.write(content)
    assert os.system(f'{sys.executable} {filename}') == 0


@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_kernel():
    x = ti.var(ti.f32, (3, 4))

    @ti.kernel
    def func():
        print(x[2, 3])

    func()

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after kernel invocation!


@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_access():
    x = ti.var(ti.f32, (3, 4))

    print(x[2, 3])

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after Python-scope tensor access!


@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_get_shape():
    x = ti.var(ti.f32, (3, 4))

    print(x.shape)

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after Python-scope tensor access!
