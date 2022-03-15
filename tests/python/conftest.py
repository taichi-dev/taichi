import pytest

import taichi as ti


@pytest.fixture(autouse=True)
def wanted_arch(req_arch, req_options):
    ti.init(arch=req_arch, enable_fallback=False, **req_options)
    yield
    ti.reset()
