import taichi as ti

from async_cases import *

rerun = False

if rerun:
    fuse_dense_x2y2z()
    fuse_reduction()
    fill_1d()
    fill_scalar()

ti.benchmark_plot(fn='benchmark.yml',
                  cases=['fuse_dense_x2y2z', 'fuse_reduction', 'fill_1d', 'fill_scalar'],
                  archs=['x64', 'cuda'],
                  bars='sync_vs_async',
                  left_margin=0.2)
