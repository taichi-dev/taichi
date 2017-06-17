import taichi as tc

if __name__ == '__main__':
    workload = 10000
    benchmark = tc.system.Benchmark('mpm_kernel', workload=workload, brute_force=False)
    benchmark.test()
    t = benchmark.run(100)
    print 'Brute Force', t
    benchmark = tc.system.Benchmark('mpm_kernel', workload=workload, brute_force=True)
    t = benchmark.run(100)
    print 'SIMD', t
