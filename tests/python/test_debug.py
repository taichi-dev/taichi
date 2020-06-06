import taichi as ti


def test_cpu_debug_snode_reader():
    ti.init(arch=ti.x64, debug=True)

    x = ti.var(ti.f32, shape=())
    x[None] = 10.0

    assert x[None] == 10.0


@ti.must_throw(RuntimeError)
def test_cpu_debug_snode_writer_out_of_bound():
    ti.init(arch=ti.x64, debug=True)
    ti.set_gdb_trigger(False)

    x = ti.var(ti.f32, shape=3)
    x[3] = 10.0


@ti.must_throw(RuntimeError)
def test_cpu_debug_snode_writer_out_of_bound_negative():
    ti.init(arch=ti.x64, debug=True)
    ti.set_gdb_trigger(False)

    x = ti.var(ti.f32, shape=3)
    x[-1] = 10.0


@ti.must_throw(RuntimeError)
def test_cpu_debug_snode_reader_out_of_bound():
    ti.init(arch=ti.x64, debug=True)
    ti.set_gdb_trigger(False)

    x = ti.var(ti.f32, shape=3)
    a = x[3]


@ti.must_throw(RuntimeError)
def test_cpu_debug_snode_reader_out_of_bound_negative():
    ti.init(arch=ti.x64, debug=True)
    ti.set_gdb_trigger(False)

    x = ti.var(ti.f32, shape=3)
    a = x[-1]


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


@ti.must_throw(RuntimeError)
def test_out_of_bound_with_offset():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32, shape=(8, 16), offset=(-8, -8))

    @ti.kernel
    def func():
        x[0, 0] = 1

    func()


def test_not_out_of_bound_with_offset():
    ti.init(debug=True)
    ti.set_gdb_trigger(False)
    x = ti.var(ti.i32, shape=(8, 16), offset=(-4, -8))

    @ti.kernel
    def func():
        x[-4, -8] = 1
        x[3, 7] = 2

    func()
