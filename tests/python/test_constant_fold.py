import taichi as ti


@ti.test(require=ti.extension.async_mode, async_mode=True)
def test_constant_fold():
    n = 100

    @ti.kernel
    def series() -> int:
        s = 0
        for i in ti.static(range(n)):
            a = i + 1
            s += a * a
        return s

    # \sum_{i=1}^n (i^2) = n * (n + 1) * (2n + 1) / 6
    expected = n * (n + 1) * (2 * n + 1) // 6
    assert series() == expected
