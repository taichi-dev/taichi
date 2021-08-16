import taichi as ti

res = (600, 400)
ti.init(arch=ti.cuda)

window = ti.ui.Window("heyy", res)

frame_id = 0
while window.running:

    window.show()
