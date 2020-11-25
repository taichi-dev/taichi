import taichi as ti

def test_listgen_opt_with_offsets():
    x = ti.field(dtype=ti.i32)

    ti.root.pointer(ti.i, 4).dense(ti.i, 4).place(x, offset=-8)

    @ti.kernel
    def inc():
        for i in x:
            x[i] += 1

    for i in range(10):
        inc()
    
    ti.sync()
    ti.core.print_stat()
    assert ti.get_kernel_stats().get_counters()['launched_tasks_list_gen'] <= 2
    
ti.init(async_mode=True, print_ir=True)
test_listgen_opt_with_offsets()
