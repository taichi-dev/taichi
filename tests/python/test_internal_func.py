import taichi as ti
import time


# TODO: these are not really tests...
def all_archs_for_this(test):
    # ti.call_internal() is not supported on CUDA, Metal, OpenGL yet
    return ti.archs_excluding(ti.metal, ti.opengl, ti.cuda)(test)


@all_archs_for_this
def test_basic():
    @ti.kernel
    def test():
        for i in range(10):
            ti.call_internal("do_nothing")

    test()


@all_archs_for_this
def test_host_polling():
    return

    @ti.kernel
    def test():
        ti.call_internal("refresh_counter")

    for i in range(10):
        print('updating tail to', i)
        test()
        time.sleep(0.1)


@all_archs_for_this
def test_list_manager():
    @ti.kernel
    def test():
        ti.call_internal("test_list_manager")

    test()
    test()


@all_archs_for_this
def test_node_manager():
    @ti.kernel
    def test():
        ti.call_internal("test_node_allocator")

    test()
    test()


@all_archs_for_this
def test_node_manager_gc():
    @ti.kernel
    def test_cpu():
        ti.call_internal("test_node_allocator_gc_cpu")

    test_cpu()
