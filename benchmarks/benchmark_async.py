import taichi as ti

from async_cases import *
from async_advection import *

rerun = True

cases = [
    chain_copy, increments, fill_array, sparse_saxpy, autodiff,
    stencil_reduction, mpm_splitted, advection_2d
]

cases = [deep_hierarchy]

if rerun:
    for c in cases:
        c()

case_names = [c.__name__ for c in cases]

ti.benchmark_plot(fn='benchmark.yml',
                  cases=case_names,
                  archs=['x64', 'cuda'],
                  bars='sync_vs_async',
                  left_margin=0.2)
