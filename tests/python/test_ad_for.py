import taichi as ti


def test_ad_pow():
    ti.init(print_ir=True)
    
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
        # assert p[i] == 3 * b[i]
        p.grad[i] = 1
    
    pow.grad()

    for i in range(N):
        print(a.grad[i])
        
test_ad_pow()
        
# TODO: test Fibonacci

# TODO: test local atomic add gradients

# TODO: test integer stack (primal without adjoint)