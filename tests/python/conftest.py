import os

import pytest


@pytest.fixture(autouse=True, params=(os.getenv("TI_WANTED_ARCHS", ""), ))
def wanted_arch(request):
    return request.param
