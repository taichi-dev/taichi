import taichi as ti
import time


def test_do_nothing():
    ti.init(arch=ti.cuda)
    
    @ti.kernel
    def test():
        for i in range(10):
            ti.call_internal("do_nothing")

    test()


def test_active_mask():
    ti.init(arch=ti.cuda)
    
    @ti.kernel
    def test():
        for i in range(48):
            if i % 2 == 0:
                ti.call_internal("test_active_mask")
    
    test()
    
def test_shfl_down():
    ti.init(arch=ti.cuda, print_kernel_nvptx=True)
    
    @ti.kernel
    def test():
        for i in range(32):
            # if i % 2 == 0:
            ti.call_internal("test_shfl")
    
    test()

test_shfl_down()
