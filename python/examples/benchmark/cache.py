import taichi as tc
import matplotlib.pyplot as plt


def analysis_working_set_size():
    tot = 25
    x, y = [], []
    for i in range(tot):
        size = 2 ** i * 32
        benchmark = tc.system.Benchmark('cache_strided_read', working_set_size=size, workload=1000000, step=10000000007,
                                        returns_time=True)
        t = benchmark.run(20)
        x.append(size)
        t = 64 / t * 1e-9
        y.append(t)
        print size, t, 'GB/s'

    plt.semilogx(x, y, basex=2)
    # plt.ylim(0, 100)
    plt.show()


def analysis_stride():
    tot = 64
    x, y = [], []
    for i in range(tot):
        step = i
        benchmark = tc.system.Benchmark('cache_strided_read', working_set_size=2 ** 20, workload=1000000, step=step)
        t = benchmark.run(100)
        x.append(step)
        y.append(t)
        print '%d, %.2f cyc' % (step, t)

    plt.plot(x, y, 'x-')
    plt.ylim(0, 7)
    plt.show()


if __name__ == '__main__':
    # analysis_stride()
    analysis_working_set_size()
