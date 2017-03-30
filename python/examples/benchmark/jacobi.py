import taichi as tc
import matplotlib.pyplot as plt
import time


def analysis_bf():
    tot = 5
    methods = [
        'bf',
        'serial'
    ]
    bits = [
        32,
        64
    ]
    for m in methods:
        for b in bits:
            ins = 'jacobi_%s_%d' % (m, b)
            x, y = [], []
            print 'Running', ins
            for i in range(tot):
                n = 2 ** i * 32
                print '%5d' % n
                benchmark = tc.system.Benchmark(ins, n=n, warm_up_iterations=0)
                assert benchmark.test()
                iterations = max(1, int(8e7 / (n ** 3)))
                t = benchmark.run(iterations)
                x.append(n)
                y.append(t)
                print '%.3f' % (t), 'cyc / ele'
            plt.semilogx(x, y, basex=2, label=ins)

    plt.legend()
    plt.ylim(0, 100)
    plt.show()


if __name__ == '__main__':
    analysis_bf()
