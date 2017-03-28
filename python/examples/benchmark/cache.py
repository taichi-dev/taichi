import taichi as tc

if __name__ == '__main__':
    for i in range(21):
        benchmark = tc.system.Benchmark('cache', working_set_size=2 ** i * 32, workload=100000, step=10000007)
        print benchmark.run(100) * 1000000, 'ns'
