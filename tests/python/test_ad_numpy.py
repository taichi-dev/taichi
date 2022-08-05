import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_to_numpy():
    a = ti.field(dtype=float, shape=(), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    def func():
        b = a.to_numpy()

    with pytest.raises(RuntimeError) as e:
        with ti.ad.Tape(loss):
            func()
    assert 'Exporting data to external array (such as numpy array) not supported in AutoDiff for now' in e.value.args[
        0]


@test_utils.test()
def test_from_numpy():
    a = ti.field(dtype=float, shape=(), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    def func():
        a.from_numpy(np.asarray(1))

    with pytest.raises(RuntimeError) as e:
        with ti.ad.Tape(loss):
            func()
    assert 'Importing data from external array (such as numpy array) not supported in AutoDiff for now' in e.value.args[
        0]
