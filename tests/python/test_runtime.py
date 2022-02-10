import copy
import os
import sys
from contextlib import contextmanager

import pytest

import taichi as ti
from tests import test_utils


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
    'log_level': ['info', ['error', 'warn', 'info', 'debug', 'trace']],
    'gdb_trigger': [False, TF],
    'advanced_optimization': [True, TF],
    'debug': [False, TF],
    'print_ir': [False, TF],
    'verbose': [True, TF],
    'fast_math': [True, TF],
    'async_mode': [False, TF],
    'flatten_if': [False, TF],
    'simplify_before_lower_access': [True, TF],
    'simplify_after_lower_access': [True, TF],
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
    'log_level',
    'gdb_trigger',
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


@pytest.mark.parametrize('arch', test_utils.expected_archs())
def test_init_arch(arch):
    with patch_os_environ_helper({}, excludes=['TI_ARCH']):
        ti.init(arch=arch)
        assert ti.cfg.arch == arch
    with patch_os_environ_helper({'TI_ARCH': ti._lib.core.arch_name(arch)},
                                 excludes=['TI_ARCH']):
        ti.init(arch=ti.cc)
        assert ti.cfg.arch == arch


def test_init_bad_arg():
    with pytest.raises(KeyError):
        ti.init(_test_mode=True, debug=True, foo_bar=233)


def test_init_require_version():
    ti_core = ti._lib.utils.import_ti_core()
    require_version = '{}.{}.{}'.format(ti_core.get_version_major(),
                                        ti_core.get_version_minor(),
                                        ti_core.get_version_patch())
    ti.init(_test_mode=True, debug=True, require_version=require_version)


def test_init_bad_require_version():
    with pytest.raises(Exception):
        ti_core = ti._lib.utils.import_ti_core()
        bad_require_version = '{}.{}.{}'.format(
            ti_core.get_version_major(), ti_core.get_version_minor(),
            ti_core.get_version_patch() + 1)
        ti.init(_test_mode=True,
                debug=True,
                require_version=bad_require_version)


@pytest.mark.parametrize(
    'level', [ti.DEBUG, ti.TRACE, ti.INFO, ti.WARN, ti.ERROR, ti.CRITICAL])
@test_utils.test()
def test_supported_log_levels(level):
    spec_cfg = ti.init(_test_mode=True, log_level=level)
    assert spec_cfg.log_level == level


@pytest.mark.parametrize(
    'level', [ti.DEBUG, ti.TRACE, ti.INFO, ti.WARN, ti.ERROR, ti.CRITICAL])
@test_utils.test()
def test_supported_log_levels(level):
    spec_cfg = ti.init(_test_mode=True)
    ti.set_logging_level(level)
    assert ti._logging.is_logging_effective(level)
