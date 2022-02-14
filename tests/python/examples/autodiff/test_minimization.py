import random

import taichi as ti
from tests import test_utils


def test_minimization():
    from taichi.examples.autodiff.minimization import (L, gradient_descent, n,
                                                       reduce, x, y)

    for i in range(n):
        x[i] = random.random()
        y[i] = random.random()

    for k in range(100):
        with ti.Tape(loss=L):
            reduce()
        gradient_descent()

    for i in range(n):
        assert x[i] == test_utils.approx(y[i], rel=1e-2)
