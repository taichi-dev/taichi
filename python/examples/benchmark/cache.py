import taichi as tc
import matplotlib.pyplot as plt


def analysis_working_set_size():
    tot = 25
    x, y = [], []
    for i in range(tot):
        size = 2 ** i * 32
        benchmark = tc.system.Benchmark('cache', working_set_size=size, workload=1000000, step=10000000007)
        t = benchmark.run(10)
        x.append(size)
        y.append(t)
        print size, t, 'cyc'

    plt.semilogx(x, y, basex=2)
    plt.ylim(0, 100)
    plt.show()


def analysis_stride():
    tot = 64
    x, y = [], []
    for i in range(tot):
        step = i
        benchmark = tc.system.Benchmark('cache', working_set_size=2 ** 20, workload=1000000, step=step)
        t = benchmark.run(100)
        x.append(step)
        y.append(t)
        print step, t, 'cyc'

    plt.plot(x, y, 'x-')
    plt.ylim(0, 7)
    plt.show()


if __name__ == '__main__':
    analysis_stride()
