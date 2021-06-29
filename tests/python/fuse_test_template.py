import time

import taichi as ti


def template_fuse_dense_x2y2z(size=1024**3,
                              repeat=10,
                              first_n=100,
                              benchmark=0,
                              benchmark_repeat=50):
    x = ti.field(ti.i32, shape=(size, ))
    y = ti.field(ti.i32, shape=(size, ))
    z = ti.field(ti.i32, shape=(size, ))
    first_n = min(first_n, size)

    @ti.kernel
    def x_to_y():
        for i in x:
            y[i] = x[i] + 1

    @ti.kernel
    def y_to_z():
        for i in x:
            z[i] = y[i] + 4

    def x_to_y_to_z():
        x_to_y()
        y_to_z()

    for i in range(first_n):
        x[i] = i * 10

    if benchmark:
        ti.benchmark(x_to_y_to_z, repeat=benchmark_repeat)
    else:
        # Simply test
        for _ in range(repeat):
            t = time.time()
            x_to_y()
            ti.sync()
            print('x_to_y', time.time() - t)

        for _ in range(repeat):
            t = time.time()
            y_to_z()
            ti.sync()
            print('y_to_z', time.time() - t)

        for _ in range(repeat):
            t = time.time()
            x_to_y_to_z()
            ti.sync()
            print('fused x->y->z', time.time() - t)

        for i in range(first_n):
            assert x[i] == i * 10
            assert y[i] == x[i] + 1
            assert z[i] == x[i] + 5


def template_fuse_reduction(size=1024**3,
                            repeat=10,
                            first_n=100,
                            benchmark=0,
                            benchmark_repeat=50):
    x = ti.field(ti.i32, shape=(size, ))
    first_n = min(first_n, size)

    @ti.kernel
    def reset():
        for i in range(first_n):
            x[i] = i * 10

    @ti.kernel
    def inc():
        for i in x:
            x[i] = x[i] + 1

    if benchmark:

        def repeated_inc():
            for _ in range(repeat):
                inc()

        ti.benchmark(repeated_inc, repeat=benchmark_repeat)
    else:
        # Simply test
        reset()
        ti.sync()
        for _ in range(repeat):
            t = time.time()
            inc()
            ti.sync()
            print('single inc', time.time() - t)

        reset()
        ti.sync()
        t = time.time()
        for _ in range(repeat):
            inc()
        ti.sync()
        duration = time.time() - t
        print(
            f'fused {repeat} inc: total={duration} average={duration / repeat}'
        )

        for i in range(first_n):
            assert x[i] == i * 10 + repeat
