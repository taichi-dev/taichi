import taichi as ti

@ti.test(arch=ti.cuda)
def test_thread_idx():
    x = ti.field(ti.i32, shape = (32,8))
    
    @ti.kernel
    def func():
        for i in range(32):
            for j in range(8):
                t = ti.thread_idx()
                x[t] += 1
    
    func()
    assert x.to_numpy().sum() == 256
