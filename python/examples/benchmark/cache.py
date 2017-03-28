import taichi as tc
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
