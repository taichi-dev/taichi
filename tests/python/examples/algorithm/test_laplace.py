import os

import pytest


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_laplace():
    from taichi.examples.algorithm.laplace import laplace, x, y

    for i in range(10):
        x[i, i + 1] = 1.0

    laplace()

    for i in range(10):
        assert y[i, i + 1] == (4.0 if i % 3 == 1 else 0.0)
