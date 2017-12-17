import taichi as tc
import matplotlib.pyplot as plt
import time


def analysis_bf():
  tot = 4
  cls_methods = {
      #'bf': [''],
      #'serial': ['relative_noif', 'relative_noif_inc']#'relative_noif_inc_unroll2', 'relative_noif_inc_unroll4']
      'serial': ['relative_noif_inc_unroll4'],
      'simd': ['sse', 'avx'],
      #'simd': ['avx', 'sse'],
      #'serial': ['relative_noif_inc_unroll']#, 'relative_noif_inc', 'relative_noif', 'relative', 'naive'],
  }
  bits = [
      #64,
      32,
  ]
  maximum = 0
  for cls, methods in list(cls_methods.items()):
    for method in methods:
      for b in bits:
        ins = 'jacobi_%s_%d' % (cls, b)
        x, y = [], []
        print('Running', ins, '(%s)' % method)
        benchmark = tc.system.Benchmark(
            ins,
            n=128,
            warm_up_iterations=0,
            iteration_method=method,
            ignore_boundary=4)
        assert benchmark.test()
        for i in range(tot):
          n = 2**i * 64
          print('%5d' % n)
          benchmark = tc.system.Benchmark(
              ins,
              n=n,
              warm_up_iterations=0,
              iteration_method=method,
              ignore_boundary=8)
          iterations = max(1, int(2e8 / (n**3)))
          t = benchmark.run(iterations)
          maximum = max(maximum, t)
          x.append(n)
          y.append(t)
          print('%.3f' % (t), 'CPE')
        plt.semilogx(
            x, y, basex=2, label=ins + ('_' + method if method else ''))
        plt.xlabel('N')
        plt.ylabel('Cycles per Element')

  plt.legend()
  plt.ylim(0, maximum * 1.3)
  plt.show()


def analysis_mt():
  cls_methods = {
      'simd': ['sse_threaded'],
  }
  bits = [
      32,
  ]
  maximum = 0
  for cls, methods in list(cls_methods.items()):
    for method in methods:
      for b in bits:
        ins = 'jacobi_%s_%d' % (cls, b)
        x, y = [], []
        print('Running', ins, '(%s)' % method)
        benchmark = tc.system.Benchmark(
            ins,
            n=128,
            warm_up_iterations=0,
            iteration_method=method,
            ignore_boundary=4)
        assert benchmark.test()
        n = 512
        print('%5d' % n)
        for i in range(4):
          num_threads = 2**i
          benchmark = tc.system.Benchmark(
              ins,
              n=n,
              warm_up_iterations=0,
              iteration_method=method,
              ignore_boundary=8)
          iterations = max(1, int(2e9 / (n**3)))
          t = benchmark.run(iterations)
          maximum = max(maximum, t)
          x.append(num_threads)
          y.append(t)
          print('%.3f' % (t), 'CPE')

        plt.semilogx(
            x, y, basex=2, label=ins + ('_' + method if method else ''))
        plt.xlabel('# Cores')
        plt.ylabel('Cycles per Element')

  plt.legend()
  plt.ylim(0, maximum * 1.3)
  plt.show()


def keep_running():
  ins = 'jacobi_simd_32'
  method = 'sse'
  benchmark = tc.system.Benchmark(
      ins,
      n=128,
      warm_up_iterations=0,
      iteration_method=method,
      ignore_boundary=4)
  assert benchmark.test()
  benchmark = tc.system.Benchmark(
      ins,
      n=512,
      warm_up_iterations=0,
      iteration_method=method,
      ignore_boundary=4)
  while True:
    print(benchmark.run(100))


if __name__ == '__main__':
  # keep_running()
  analysis_mt()
