import os
import random

import pytest

import taichi as ti


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_minimization():
    from taichi.examples.autodiff.minimization import (L, gradient_descent, n,
                                                       reduce, x, y)

    for i in range(n):
        x[i] = random.random()
        y[i] = random.random()

    for k in range(100):
        with ti.ad.Tape(loss=L):
            reduce()
        gradient_descent()

    for i in range(n):
        assert x[i] == pytest.approx(y[i], rel=1e-2)
