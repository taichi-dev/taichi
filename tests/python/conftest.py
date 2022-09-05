import os

import pytest

import taichi as ti


@pytest.fixture(autouse=True)
def wanted_arch(req_arch, req_options):
    if req_arch is not None:
        if not {'device_memory_GB', 'device_memory_fraction'
                } & set(req_options):
            # Lower GPU requirements
            req_options['device_memory_GB'] = 0.5

        ti.init(arch=req_arch, enable_fallback=False, **req_options)
    yield
    if req_arch is not None:
        ti.reset()


def pytest_generate_tests(metafunc):
    if not getattr(metafunc.function, '__ti_test__', False):
        # For test functions not wrapped with @test_utils.test(),
        # fill with empty values to avoid undefined fixtures
        metafunc.parametrize('req_arch,req_options', [(None, None)],
                             ids=['none'])
