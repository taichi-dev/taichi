import taichi as tc

if __name__ == '__main__':
    benchmark = tc.system.Benchmark('cache')
    print benchmark.run(100)

