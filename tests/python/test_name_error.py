import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_name_error():
    with pytest.raises(ti.TaichiNameError, match='Name "a" is not defined'):

        @ti.kernel
        def foo():
            a + 1

        foo()
