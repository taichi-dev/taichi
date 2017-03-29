import taichi as tc
import matplotlib.pyplot as plt
import time


def analysis_bf():
    tot = 5
    cls_methods = {
        'bf': [''],
        'serial': ['relative_noif_inc_unroll', 'relative_noif_inc',
                   'relative_noif',
                   'relative', 'naive'
                   ]
    }
    bits = [
        32,
        64
    ]
    for cls, methods in cls_methods.iteritems():
        for method in methods:
            for b in bits:
                ins = 'jacobi_%s_%d' % (cls, b)
                x, y = [], []
                print 'Running', ins, '(%s)' % method
                benchmark = tc.system.Benchmark(ins, n=128, warm_up_iterations=0, iteration_method=method)
                assert benchmark.test()
                for i in range(tot):
                    n = 2 ** i * 32
                    print '%5d' % n
                    benchmark = tc.system.Benchmark(ins, n=n, warm_up_iterations=0, iteration_method=method)
                    iterations = max(1, int(8e7 / (n ** 3)))
                    t = benchmark.run(iterations)
                    x.append(n)
                    y.append(t)
                    print '%.3f' % (t), 'CPE'
                plt.semilogx(x, y, basex=2, label=ins + ('_' + method if method else ''))

    plt.legend()
    plt.ylim(0, 100)
    plt.show()


if __name__ == '__main__':
    analysis_bf()
