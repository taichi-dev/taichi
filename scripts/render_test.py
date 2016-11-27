from renderer.render import *

def create_object(name, x, y=0, z=0, s=1, material='wall'):
    mesh = tc.create_mesh()
    mesh.initialize(P(filename='../assets/meshes/%s.obj' % name))
    mesh.set_material(assets.materials.get_material(material))
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    return mesh

def create_light(t):
    mesh = tc.create_mesh()
    mesh.initialize(P(filename='../assets/meshes/plane.obj'))
    material = tc.create_surface_material('emissive')
    e = 0.00001
    material.initialize(P(color=(e, e, e)))
    mesh.set_material(material)
    mesh.translate(Vector(math.cos(t) * -3, 5, -1))
    mesh.scale_s(0.00001)
    mesh.rotate_euler(Vector(0, 0, 180 + math.cos(t) * 45))
    return mesh

def render_frame(i, t):
    width, height = 960, 540
    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=60,
                    origin=(0, 3, 9), look_at=(0, 0, 0), up=(0, 1, 0))

    renderer = Renderer('pt', '../output/frames/%d.png' % i)
    renderer.initialize(width=width, height=height, min_path_length=1, max_path_length=10,
                        initial_radius=0.05, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1)
    renderer.set_camera(camera.c)

    air = tc.create_volume_material("vacuum")
    air.initialize(P(scattering=0.01))

    scene = tc.create_scene()
    scene.set_atmosphere_material(air)

    scene.add_mesh(create_object('suzanne', 2.5, material='gold'))
    scene.add_mesh(create_object('cube', 0, material='glossy'))
    scene.add_mesh(create_object('sphere', -2.5, material='glass'))
    scene.add_mesh(create_object('holder', 0, -1, -4, 2, material='wall'))
    #scene.add_mesh(create_object('plane', 0, -2, 0, 2, material='wall'))
    scene.add_mesh(create_light(0))

    envmap = tc.create_environment_map('base')
    envmap.initialize(P(filepath='c:/tmp/20060807_wells6_hd.hdr'))
    #envmap.initialize(P(filepath='c:/tmp/schoenbrunn-front_hd.hdr'))

    scene.set_envmap(envmap)

    scene.finalize()

    renderer.set_scene(scene)
    renderer.render(10000000)

if __name__ == '__main__':

    frames = 20
    for i in [0]:#range(frames):
        render_frame(i, 2 * math.pi * i / frames)
    from tools import video
    video.make_video('../output/frames/%d.png', 960, 540, '../output/video.mp4')
