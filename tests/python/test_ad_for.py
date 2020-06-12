import taichi as ti


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_pow():
    N = 10
    a = ti.var(ti.f32, shape=N, needs_grad=True)
    b = ti.var(ti.i32, shape=N)
    p = ti.var(ti.f32, shape=N, needs_grad=True)
    
    @ti.kernel
    def pow():
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret * a[i]
            p[i] = ret
            
    for i in range(N):
        a[i] = 3
        b[i] = i
        
    pow()
    
    for i in range(N):
        assert p[i] == 3 ** b[i]
        p.grad[i] = 1
    
    pow.grad()

    for i in range(N):
        assert a.grad[i] == b[i] * 3 ** (b[i] - 1)
        
# TODO: test Fibonacci

# TODO: test local atomic add gradients

# TODO: test integer stack (primal without adjoint)