import taichi as ti

ti.init(print_preprocessed=True, print_ir=True)

@ti.kernel
def test_cpu():
    ret = ti.call_internal("test_internal_func_args", 1.0, 2.0, 3)
    # ret = ti.call_internal("test_internal_func_args", 1.0, 2.0, 3.0)
    print(ret)

test_cpu()
