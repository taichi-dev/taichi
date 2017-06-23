import taichi as tc
import time

if __name__ == '__main__':
    workload = 10000
    benchmark = tc.system.Benchmark('spgrid', workload=workload, brute_force=True)
    print 'Check memory usage'
    benchmark.test()
    t = benchmark.run(1)
