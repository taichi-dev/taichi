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


@ti.host_arch_only
def test_static_assert_is_static():
    @ti.kernel
    def func():
        x = 0
        ti.static_assert(x)  # Expr is not None

    func()


@ti.host_arch_only
@ti.must_throw(AssertionError)
def test_static_assert_message():
    x = 3

    @ti.kernel
    def func():
        ti.static_assert(x == 4, "Oh, no!")

    func()


@ti.host_arch_only
def test_static_assert_vector_n_ok():
    x = ti.Vector(4, ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.n == 4)

    func()


@ti.host_arch_only
def test_static_assert_data_type_ok():
    x = ti.var(ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.data_type() == ti.f32)

    func()
