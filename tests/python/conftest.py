import os

import pytest

import taichi as ti


@pytest.fixture(autouse=True)
def wanted_arch(req_arch, req_options):
    if req_arch is not None:
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


IS_WORKER = False


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    global IS_WORKER
    IS_WORKER = hasattr(config, "workerinput")


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    '''
    Intentionally crash test workers when a test fails.
    This is to avoid the failing test leaving a corrupted GPU state for the
    following tests.
    '''
    if not IS_WORKER:
        return

    if report.outcome not in ('error', 'failed'):
        return

    os._exit(0)
