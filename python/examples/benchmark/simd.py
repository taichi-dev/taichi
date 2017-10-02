import taichi as tc
'''
Result:
Brute Force 5.35472679138
SIMD 4.87170886993
'''

if __name__ == '__main__':
  workload = 10000000
  benchmark = tc.system.Benchmark(
      'matrix4s', workload=workload, brute_force=True)
  benchmark.test()
  t = benchmark.run(100)
  print('Brute Force', t)
  benchmark = tc.system.Benchmark(
      'matrix4s', workload=workload, brute_force=False)
  t = benchmark.run(100)
  print('SIMD', t)
