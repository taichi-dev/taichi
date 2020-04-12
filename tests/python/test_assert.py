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


@ti.must_throw(RuntimeError)
def test_out_of_bound():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[3, 16] = 1

    func()


def test_not_out_of_bound():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[7, 15] = 1

    func()


@ti.must_throw(RuntimeError)
def test_out_of_bound_dynamic():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32)

    ti.root.dynamic(ti.i, 16, 4).place(x)

    @ti.kernel
    def func():
        x[17] = 1

    func()


def test_not_out_of_bound_dynamic():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32)

    ti.root.dynamic(ti.i, 16, 4).place(x)

    @ti.kernel
    def func():
        x[3] = 1

    func()
