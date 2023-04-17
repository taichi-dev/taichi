import sys

import pytest

# rerunfailures use xdist version number to determine if it is compatible
# but we are using a forked version of xdist(with git hash as it's version),
# so we need to override it
import pytest_rerunfailures

import taichi as ti

pytest_rerunfailures.works_with_current_xdist = lambda: True


@pytest.fixture(autouse=True)
def wanted_arch(request, req_arch, req_options):
    if req_arch is not None:
        if req_arch == ti.cuda:
            if not request.node.get_closest_marker("run_in_serial"):
                # Optimization only apply to non-serial tests, since serial tests
                # are picked out exactly because of extensive resource consumption.
                # Separation of serial/non-serial tests is done by the test runner
                # through `-m run_in_serial` / `-m not run_in_serial`.
                req_options = {
                    "device_memory_GB": 0.3,
                    "cuda_stack_limit": 1024,
                    **req_options,
                }
            else:
                # Serial tests run without aggressive resource optimization
                req_options = {"device_memory_GB": 1, **req_options}
        ti.init(arch=req_arch, enable_fallback=False, **req_options)
    yield
    if req_arch is not None:
        ti.reset()


def pytest_generate_tests(metafunc):
    if not getattr(metafunc.function, "__ti_test__", False):
        # For test functions not wrapped with @test_utils.test(),
        # fill with empty values to avoid undefined fixtures
        metafunc.parametrize("req_arch,req_options", [(None, None)], ids=["none"])


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """
    Intentionally crash test workers when a test fails.
    This is to avoid the failing test leaving a corrupted GPU state for the
    following tests.
    """

    interactor = getattr(sys, "xdist_interactor", None)
    if not interactor:
        # not running under xdist, or xdist is not active,
        # or using stock xdist (we need a customized version)
        return

    if report.outcome not in ("rerun", "error", "failed"):
        return

    layoff = False

    for _, loc, _ in report.longrepr.chain:
        if "CUDA_ERROR_OUT_OF_MEMORY" in loc.message:
            layoff = True
            break

    interactor.retire(layoff=layoff)
