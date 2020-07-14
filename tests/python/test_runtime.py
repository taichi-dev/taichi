import taichi as ti
import sys, os, copy
from contextlib import contextmanager
import pytest


@contextmanager
def patch_os_environ_helper(custom_environ: dict, excludes: dict):
    """
    Temporarily patch os.environ for testing.
    Originally created by @rexwangcc in test_cli.py
    @archibate tweaked this method to be an os.environ patcher.

    The patched environ will be:
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
    'excepthook': [False, TF],
    'advanced_optimization': [True, TF],
    'debug': [False, TF],
    'print_ir': [False, TF],
    'verbose': [True, TF],
    'fast_math': [True, TF],
    'async': [False, TF],
    'flatten_if': [False, TF],
    'simplify_before_lower_access': [True, TF],
    'simplify_after_lower_access': [True, TF],
    'use_unified_memory': [ti.get_os_name() != 'win', TF],
    'print_benchmark_stat': [False, TF],
    'kernel_profiler': [False, TF],
    'check_out_of_bound': [False, TF],
    'print_accessor_ir': [False, TF],
    'print_evaluator_ir': [False, TF],
    'print_struct_llvm_ir': [False, TF],
    'print_kernel_llvm_ir': [False, TF],
    'print_kernel_llvm_ir_optimized': [False, TF],
    # FIXME: figure out why these two failed test:
    #'device_memory_fraction': [0.0, [0.5, 1, 0]],
    #'device_memory_GB': [1.0, [0.5, 1, 1.5, 2]],
}

env_configs = ['TI_' + key.upper() for key in init_args.keys()]

special_init_cfgs = [
    'print_preprocessed',
    'log_level',
    'gdb_trigger',
    'excepthook',
]


@pytest.mark.parametrize('key,values', init_args.items())
def test_init_arg(key, values):
    default, values = values

    # helper function:
    def test_arg(key, value, kwargs={}):
        spec_cfg = ti.init(_test_mode=True, **kwargs)
        if key in special_init_cfgs:
            cfg = spec_cfg
        else:
            cfg = ti.cfg
        assert getattr(cfg, key) == value

    with patch_os_environ_helper({}, excludes=env_configs):
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
        with patch_os_environ_helper(environ, excludes=env_configs):
            test_arg(key, value)


@ti.must_throw(KeyError)
def test_init_bad_arg():
    ti.init(_test_mode=True, debug=True, foo_bar=233)


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
