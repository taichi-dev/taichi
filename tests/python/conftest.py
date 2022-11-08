import os

import pytest

import taichi as ti


@pytest.fixture(autouse=True)
def wanted_arch(request, req_arch, req_options):
    if req_arch is not None:
        if req_arch == ti.cuda:
            if not request.node.get_closest_marker('run_in_serial'):
                # Optimization only apply to non-serial tests, since serial tests
                # are picked out exactly because of extensive resource consumption.
                # Separation of serial/non-serial tests is done by the test runner
                # through `-m run_in_serial` / `-m not run_in_serial`.
                req_options = {
                    'default_gpu_block_dim': 32,
                    'saturating_grid_dim': 8,
                    'device_memory_GB': 0.4,
                    **req_options
                }
            else:
                # Serial tests run without aggressive resource optimization
                req_options = {'device_memory_GB': 1, **req_options}
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
    config.addinivalue_line(
        "markers",
        "run_in_serial: mark test to run serially(usually for resource intensive tests)."
    )

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

    if report.outcome not in ('rerun', 'error', 'failed'):
        return

    os._exit(0)
