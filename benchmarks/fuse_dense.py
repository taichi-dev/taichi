import taichi as ti
import os
import sys

sys.path.append(os.path.join(ti.core.get_repo_dir(), 'tests', 'python'))

from fuse_test_template import template_fuse_dense_x2y2z, \
    template_fuse_reduction


@ti.archs_excluding(ti.opengl, ti.cuda)
def benchmark_fuse_dense_x2y2z():
    template_fuse_dense_x2y2z(size=10 * 1024**2,
                              repeat=10,
                              benchmark_repeat=50,
                              benchmark=True)


@ti.all_archs
def benchmark_fuse_reduction():
    template_fuse_reduction(size=10 * 1024**2,
                            repeat=10,
                            benchmark_repeat=50,
                            benchmark=True)
