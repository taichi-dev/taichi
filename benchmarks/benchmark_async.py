from async_advection import *
from async_cases import *

import taichi as ti

rerun = True

cases = [
    chain_copy, increments, fill_array, sparse_saxpy, autodiff,
    stencil_reduction, mpm_splitted, simple_advection, multires, deep_hierarchy
]

if rerun:
    for c in cases:
        print('*' * 30)
        print(f'* Running {c.__name__}')
        print('*' * 30)
        c()

case_names = [c.__name__ for c in cases]

ti.benchmark_plot(fn='benchmark.yml',
                  cases=case_names,
                  columns=[
                      'wall_clk_t', 'exec_t', 'launched_tasks',
                      'compiled_inst', 'compiled_tasks'
                  ],
                  column_titles=[
                      'Wall-clock time', 'Backend time', 'Tasks launched',
                      'Instructions emitted', 'Tasks compiled'
                  ],
                  archs=['cuda', 'x64'],
                  title='Whole-Program Optimization Microbenchmarks',
                  bars='sync_vs_async',
                  left_margin=0.2,
                  size=(11.5, 9))
