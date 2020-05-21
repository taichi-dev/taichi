import taichi as ti


@ti.must_throw(RuntimeError)
def test_assert_minimal():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)

    @ti.kernel
    def func():
        assert 0

    func()


@ti.must_throw(RuntimeError)
def test_assert_basic():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)

    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20

    func()


def test_assert_ok():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)

    @ti.kernel
    def func():
        x = 20
        assert 10 <= x <= 20

    func()
