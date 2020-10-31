import math
import time

import numpy as np
import taichi as ti

from mpm_solver import MPMSolver

write_to_disk = False

ti.init(arch=ti.cpu, async_mode=True)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41,
             show_gui=False)

mpm = MPMSolver(res=(128, 128))

for i in range(3):
    mpm.add_cube(lower_corner=[0.2 + i * 0.1, 0.3 + i * 0.1],
                 cube_size=[0.1, 0.1],
                 material=MPMSolver.material_elastic)

ti.sync()
t = time.time()
for frame in range(50):
    mpm.step(8e-3, print_stat=False)
    if frame < 500:
        mpm.add_cube(lower_corner=[0.1, 0.8],
                     cube_size=[0.01, 0.05],
                     velocity=[1, 0],
                     material=MPMSolver.material_sand)
    if 10 < frame < 100:
        mpm.add_cube(lower_corner=[0.6, 0.7],
                     cube_size=[0.2, 0.01],
                     material=MPMSolver.material_water,
                     velocity=[math.sin(frame * 0.1), 0])
    if 120 < frame < 200 and frame % 10 == 0:
        mpm.add_cube(
            lower_corner=[0.4 + frame * 0.001, 0.6 + frame // 40 * 0.02],
            cube_size=[0.2, 0.1],
            velocity=[-3, -1],
            material=MPMSolver.material_snow)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    # particles = mpm.particle_info()
    # gui.circles(particles['position'],
    #             radius=1.5,
    #             color=colors[particles['material']])
    # gui.show(f'{frame:06d}.png' if write_to_disk else None)

ti.sync()
t = time.time() - t
ti.print_profile_info()
ti.core.print_stat()
print(t, 's')
