import taichi as ti
import time

ti.init(kernel_profiler=True,arch=ti.cpu) #log_level=ti.TRACE

test_archs = [ti.cuda,ti.cpu]
test_dsize = [ 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824 ]
           # [  4KB,  16KB,  64KB,  256KB,     1MB,     4MB,     16MB,     64MB,     256MB,     1024MB ]    
test_dtype = [ti.i32, ti.i64, ti.f32, ti.f64]
dtype_size = {
    ti.i32: 4,
    ti.i64: 8,
    ti.f32: 4,
    ti.f64: 8
}


def performance_reduction(dtype,dsize,repeat=100):

    n = dsize//dtype_size[dtype]
    if dsize<= 4*1024*1024:
        repeat = repeat*10

    
    ## fill x
    x = ti.field(dtype, shape=n)

    if dtype in [ti.f32, ti.f64]:
        @ti.kernel
        def fill_const(n: ti.i32):
            for i in range(n):
                x[i] = 0.1
    else:
        @ti.kernel
        def fill_const(n: ti.i32):
            for i in range(n):
                x[i] = 1

    # compile the kernel first
    fill_const(n)  
    ti.sync()
    ti.kernel_profiler_clear()
    ti.sync()
    for i in range(repeat):
        fill_const(n)
    ti.sync()
    kernelname = "fill_const"
    suffix = "_c"
    quering_result = ti.kernel_profiler_query(kernelname + suffix)
    print("[query_min_time_in_ms][kernel: " + kernelname + "] = " , quering_result.min)


    ## reduce
    y = ti.field(dtype, shape=())
    if dtype in [ti.f32, ti.f64]:
        y[None] = 0.0
    else:
        y[None] = 0

    @ti.kernel
    def reduction(n: ti.i32):
        for i in range(n):
            y[None] += ti.atomic_add(y[None], x[i])

    # compile the kernel first
    reduction(n)
    ti.sync()
    ti.kernel_profiler_clear()
    ti.sync()
    for i in range(repeat):
        reduction(n)
    ti.sync()
    kernelname = reduction.__name__
    suffix = "_c"
    quering_result = ti.kernel_profiler_query(kernelname + suffix)
    print("[query_min_time_in_ms][kernel: " + kernelname + "] = " , quering_result.min)
    print("y=",y[None])



def performance():
    for backend in test_archs:
        for dtype in test_dtype:
            ti.reset()
            ti.init(kernel_profiler=True,arch=backend)
            for size in test_dsize:
                print("##################### TEST data size = %4.4f MB #####################" %(size/1024.0/1024.0))
                performance_reduction(dtype,size,100)
                time.sleep(1)



performance()


