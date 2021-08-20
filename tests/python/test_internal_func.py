import time

import taichi as ti


# TODO: these are not really tests...
def all_archs_for_this(test):
    # ti.call_internal() is not supported on CUDA, Metal, OpenGL yet
    return ti.archs_excluding(ti.metal, ti.opengl, ti.cuda, ti.vulkan)(test)


@ti.test(exclude=[ti.metal, ti.opengl, ti.cuda, ti.vulkan])
def test_basic():
    @ti.kernel
    def test():
        for _ in range(10):
            ti.call_internal("do_nothing")

    test()


@ti.test(exclude=[ti.metal, ti.opengl, ti.cuda, ti.vulkan])
def test_host_polling():
    return

    @ti.kernel
    def test():
        ti.call_internal("refresh_counter")

    for i in range(10):
        print('updating tail to', i)
        test()
        time.sleep(0.1)


@ti.test(exclude=[ti.metal, ti.opengl, ti.cuda, ti.vulkan])
def test_list_manager():
    @ti.kernel
    def test():
        ti.call_internal("test_list_manager")

    test()
    test()


@ti.test(exclude=[ti.metal, ti.opengl, ti.cuda, ti.vulkan])
def test_node_manager():
    @ti.kernel
    def test():
        ti.call_internal("test_node_allocator")

    test()
    test()


@ti.test(exclude=[ti.metal, ti.opengl, ti.cuda, ti.vulkan])
def test_node_manager_gc():
    @ti.kernel
    def test_cpu():
        ti.call_internal("test_node_allocator_gc_cpu")

    test_cpu()


@ti.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_return():
    @ti.kernel
    def test_cpu():
        ret = ti.call_internal("test_internal_func_args", 1.0, 2.0, 3)
        assert ret == 9

    test_cpu()
