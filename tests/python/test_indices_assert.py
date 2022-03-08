import platform

import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Too much virtual memory for github windows env.")
@test_utils.test(debug=True, gdb_trigger=False, packed=False, arch=[ti.cpu])
def test_indices_assert():

    overflow = ti.field(ti.u8, (3, 715827883))

    @ti.kernel
    def access_overflow():
        overflow[0, 5] = 10
        print(overflow[2, 715827882])

    with pytest.raises(RuntimeError,
                       match='The indices provided are too big!'):
        access_overflow()
