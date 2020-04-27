import taichi as ti


@ti.must_throw(RuntimeError)
def test_out_of_bound():
    ti.init(debug=True)
    ti.set_gdb_trigger(True)
    x = ti.var(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[3, 16] = 1

    func()


@ti.must_throw(RuntimeError)
def test_out_of_bound_dynamic():
    ti.init(debug=True)
    ti.set_gdb_trigger(True)
    x = ti.var(ti.i32, shape=(8, 16))

    @ti.kernel
    def func():
        x[3, 16] = 1

    func()
