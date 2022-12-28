import pytest

from tests import test_utils


@pytest.mark.parametrize('i', [1, 2])
@test_utils.test()
def test_a(i):
    pass


@pytest.mark.parametrize('i', [1, 2])
@test_utils.test()
def test_b(i):
    pass


@pytest.mark.parametrize('i', [1, 2])
@test_utils.test()
def test_c(i):
    pass
