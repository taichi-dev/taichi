import taichi as ti


@ti.must_throw(IndexError)
@ti.host_arch_only
def test_struct_for_mismatch():
    x = ti.field(ti.f32, (3, 4))

    @ti.kernel
    def func():
        for i in x:
            print(i)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def test_struct_for_mismatch2():
    x = ti.field(ti.f32, (3, 4))

    @ti.kernel
    def func():
        for i, j, k in x:
            print(i, j, k)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def _test_grouped_struct_for_mismatch():
    # doesn't work for now
    # need grouped refactor
    # for now, it just throw a unfriendly message:
    # AssertionError: __getitem__ cannot be called in Python-scope
    x = ti.field(ti.f32, (3, 4))

    @ti.kernel
    def func():
        for i, j in ti.grouped(x):
            print(i, j)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def _test_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange refactor
    @ti.kernel
    def func():
        for i in ti.ndrange(3, 4):
            print(i)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def _test_ndrange_for_mismatch2():
    # doesn't work for now
    # need ndrange and grouped refactor
    @ti.kernel
    def func():
        for i, j, k in ti.ndrange(3, 4):
            print(i, j, k)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def _test_grouped_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange and grouped refactor
    @ti.kernel
    def func():
        for i in ti.grouped(ti.ndrange(3, 4)):
            print(i)

    func()


@ti.must_throw(IndexError)
@ti.host_arch_only
def _test_static_ndrange_for_mismatch():
    # doesn't work for now
    # need ndrange and static refactor
    @ti.kernel
    def func():
        for i in ti.static(ti.ndrange(3, 4)):
            print(i)

    func()
