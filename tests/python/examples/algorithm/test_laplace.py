from taichi.examples.algorithm.laplace import x, y, laplace


def test_laplace():
    for i in range(10):
        x[i, i + 1] = 1.0

    laplace()

    for i in range(10):
        assert y[i, i + 1] == (4.0 if i % 3 == 1 else 0.0)
