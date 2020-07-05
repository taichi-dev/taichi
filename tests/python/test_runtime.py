import taichi as ti
from taichi import make_temp_file
import sys, os, copy
from contextlib import contextmanager
import pytest


@contextmanager
def patch_os_environ_helper(custom_environ: dict, excludes: dict):
    """
    Temporarily patch os.environ for testing.
    Originally created by @rexwangcc in test_cli.py
    @archibate tweaked this method to be an os.environ patcher.

    The patched environ environ will be:
        custom_environ + (os.environ - excludes - custom_environ).

    I.e.:

    1. custom_environ could override os.environ.
    2. os.environ keys match excludes will not be included.

    :parameter custom_environ:
        Specify the base environment of patch, these values must
        be included.

    :parameter excludes:
        When copying from os.environ, specify keys to be excluded.
    """
    environ = {}
    for key in os.environ.keys():
        if key not in excludes:
            environ[key] = os.environ[key]
    for key in custom_environ.keys():
        environ[key] = custom_environ[key]
    try:
        cached_environ = os.environ
        os.environ = custom_environ
        yield os.environ
    finally:
        os.environ = cached_environ


TF = [True, False]
init_args = {
    # 'key': [default, choices],
    'print_preprocessed': [False, TF],
    'log_level': ['info', ['error', 'warn', 'info', 'debug', 'trace']],
    'gdb_trigger': [False, TF],
    'advanced_optimization': [True, TF],
    'debug': [False, TF],
    'print_ir': [False, TF],
    'verbose': [False, TF],
    'fast_math': [False, TF],
    'async': [False, TF],
    'use_unified_memory': [True, TF],
    'print_benchmark_stat': [False, TF],
    # FIXME: figure out why these two failed test:
    #'device_memory_fraction': [0.5, [0.5, 1, 0]],
    #'device_memory_GB': [1.0, [0.5, 1, 1.5, 2]],
}

env_configs = ['TI_' + key.upper() for key in init_args.keys()]

special_init_args = [
    'print_preprocessed',
    'log_level',
    'gdb_trigger',
    'advanced_optimization',
]


@pytest.mark.parametrize('key,values', init_args.items())
def test_init_arg(key, values):
    default, values = values

    # helper function:
    def test_arg(key, value, kwargs={}):
        args = ti.init(_test_mode=True, **kwargs)
        if key in special_init_args:
            assert args[key] == value
        else:
            assert getattr(ti.cfg, key) == value

    # test if default value is correct:
    test_arg(key, default)

    # test if specified in argument:
    for value in values:
        kwargs = {key: value}
        test_arg(key, value, kwargs)

    # test if specified in environment:
    env_key = 'TI_' + key.upper()
    for value in values:
        env_value = str(int(value) if isinstance(value, bool) else value)
        environ = {env_key: env_value}
        with patch_os_environ_helper(environ, env_configs):
            test_arg(key, value)


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
