import taichi as ti

'''
@ti.all_archs
def test_ad_if_simple():
    x = ti.var(ti.f32, shape=())
    y = ti.var(ti.f32, shape=())
    
    ti.root.lazy_grad()
    
    @ti.kernel
    def func():
        if x[None] > 0.:
            y[None] = x[None]
    
    x[None] = 1
    y.grad[None] = 1
    
    func()
    func.grad()
    
    assert x.grad[None] == 1


@ti.all_archs
def test_ad_if():
    x = ti.var(ti.f32, shape=2)
    y = ti.var(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func(i: ti.i32):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]
            

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0)
    func.grad(0)
    func(1)
    func.grad(1)

    assert x.grad[0] == 2
    assert x.grad[1] == 1

@ti.all_archs
def test_ad_if_mutable():
    x = ti.var(ti.f32, shape=2)
    y = ti.var(ti.f32, shape=2)
    
    ti.root.lazy_grad()
    
    @ti.kernel
    def func(i: ti.i32):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t
    
    
    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1
    
    func(0)
    func.grad(0)
    func(1)
    func.grad(1)
    
    assert x.grad[0] == 2
    assert x.grad[1] == 1
'''

@ti.all_archs
def test_ad_if_prallel():
    x = ti.var(ti.f32, shape=2)
    y = ti.var(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 2
    assert x.grad[1] == 1
