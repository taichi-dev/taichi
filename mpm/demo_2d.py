import math
import time

import numpy as np
import taichi as ti

from mpm_solver import MPMSolver

write_to_disk = False

ti.init(arch=ti.gpu, async_mode=False, debug=True)
        # async_opt_intermediate_file="mpm")  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41,
             show_gui=False)

mpm = MPMSolver(res=(128, 128))

ti.sync()
t = time.time()
mpm.step(8e-3)

