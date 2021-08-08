import pytest

import taichi as ti


@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True, gdb_trigger=False)
def test_assert_minimal():
    ti.set_gdb_trigger(False)

    @ti.kernel
    def func():
        assert 0

    @ti.kernel
    def func2():
        assert False

    with pytest.raises(RuntimeError):
        func()
    with pytest.raises(RuntimeError):
        func2()


@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True, gdb_trigger=False)
def test_assert_basic():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20

    with pytest.raises(RuntimeError):
        func()


@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True, gdb_trigger=False)
def test_assert_message():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20, 'Foo bar'

    with pytest.raises(RuntimeError, match='Foo bar'):
        func()


@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, 'x[%d] expect=%d got=%d' % (i, 0, x[i])

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, 'y = %f' % y

    with pytest.raises(RuntimeError, match=r'x\[10\] expect=0 got=42'):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(RuntimeError, match=r'y = 0.5'):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True, gdb_trigger=False)
def test_assert_ok():
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
    x = ti.Vector.field(4, ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.n == 4)

    func()


@ti.host_arch_only
def test_static_assert_data_type_ok():
    x = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.dtype == ti.f32)

    func()
