import taichi as ti
import math

gui = ti.GUI("Test GUI", (512, 512), background_color=0xFFFFFF)
t = 0

while True:
    t += 0.05
    gui.line((0.3, 0.3), (0.6, 0.6), radius=2, color=0x000000)
    gui.circle((0.5 + math.cos(t) * 0.1, 0.5), radius=10, color=0xFF0000)
    gui.triangle((0.25, 0.2), (0.4, 0.2), (0.4, 0.4 + math.sin(t) * 0.1),
                 color=0x00FF00)
    gui.show()
