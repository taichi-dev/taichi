from implicit_mass_string import *

window = ti.ui.Window("Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.8, 1., 0.8))
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(0.0, 0.0, 10.)
camera.lookat(0.0, 0.0, 0.)


# mesh = NODE("dragon")
# x = mesh.vertices

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

    scene.point_light(pos=(0., 10., 20.), color=(1., 1., 1.))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.set_camera(camera)

    substep()

    scene.mesh(x, indices=mesh.indices, color=(0.5, 0.5, 0.5), two_sided=True)
    # scene.mesh(mesh.vertices, indices=mesh.indices, color=(0.5, 0.5, 0.5), two_sided=True)
    canvas.scene(scene)
    window.show()
