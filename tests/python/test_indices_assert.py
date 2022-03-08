import platform

import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Too much virtual memory for github windows env.")
@test_utils.test(debug=True, gdb_trigger=False, packed=False, arch=[ti.cpu])
def test_indices_assert():

    overflow = ti.field(ti.f16, (3, 1073741824))

    @ti.kernel
    def access_overflow():
        overflow[0, 0] = 10
        print(overflow[2, 1073741823])

    with pytest.raises(RuntimeError,
                       match='The indices provided are too big!'):
        access_overflow()
