import taichi as ti

ti.benchmark_plot(fn='benchmark.yml',
                  cases=['fill_scalar', 'fuse_dense_x2y2z', 'fuse_reduction'],
                  archs=['x64'],
                  bars='sync_vs_async',
                  left_margin=0.2)
ti.benchmark_plot(fn='benchmark.yml',
                  cases=['fill_scalar', 'fuse_dense_x2y2z', 'fuse_reduction'],
                  bars='sync_regression',
                  left_margin=0.2)
