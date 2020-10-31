import taichi as ti

from mpm_solver import MPMSolver

ti.init(arch=ti.gpu, async_mode=False, debug=True)

mpm = MPMSolver(res=(128, 128))
mpm.step(8e-3)

