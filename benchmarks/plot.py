import taichi as ti
import os

from fuse_dense import benchmark_fuse_dense_x2y2z

os.environ['TI_CURRENT_BENCHMARK'] = 'fuse_dense_x2y2z'
benchmark_fuse_dense_x2y2z()

ti.benchmark_plot(fn='benchmark.yml',
                  cases=['fuse_dense_x2y2z', 'fuse_dense_x2y2z'],
                  archs=['x64'],
                  bars='sync_vs_async',
                  left_margin=0.2)
