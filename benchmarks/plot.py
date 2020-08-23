import taichi as ti

from async_cases import fuse_dense_x2y2z, fuse_reduction

fuse_dense_x2y2z()
fuse_reduction()

ti.benchmark_plot(fn='benchmark.yml',
                  cases=['fuse_dense_x2y2z', 'fuse_reduction'],
                  archs=['x64'],
                  bars='sync_vs_async',
                  left_margin=0.2)
