from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
import math

def map_filename(name):
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = name
    return filename

def create_object(name, x, y=0, z=0, s=1, r=(0, 0, 0), material='wall'):
    mesh = tc_core.create_mesh()
    mesh.initialize(P(filename=map_filename(name)))
    mesh.set_material(assets.materials.get_material(material))
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    mesh.rotate_euler(Vector(*r))
    return mesh

def create_holder(name, x, y=0, z=0, s=1, r=(0, 0, 0), t=0, taichi_s=0.9):
    rep = Texture.create_taichi_wallpaper(20, rotation=t, scale=taichi_s)
    diff = (0.1 * rep).rasterize(1024)
    spec = (0.6 * rep).rasterize(1024)
    material = SurfaceMaterial('pbr', diffuse_map=diff.id, specular_map=spec.id, glossiness=300)

    mesh = Mesh(map_filename(name), material)
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    mesh.rotate_euler(Vector(*r))
    return mesh

def create_fractal(scene):
    n = 5
    s = 0.6
    for i in range(n):
        for j in range(n):
            mesh = create_object('cube', (i - (n - 1) / 2.0) * s, j * s - 0.5, 0, s * 0.3, r=(i * 10, j * 10, 0), material='glass')
            scene.add_mesh(mesh)

def create_light(t):
    e = 1
    material = SurfaceMaterial('emissive', color=(e, e, e))
    mesh = Mesh('../assets/meshes/plane.obj', material)
    mesh.translate(Vector(math.cos(t) * -3, 5, -1))
    mesh.scale_s(1)
    mesh.rotate_euler(Vector(0, 0, 180 + math.cos(t) * 45))
    return mesh

def render_frame(i, t):
    downsample = 2
    width, height = 960 / downsample, 540 / downsample
    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=60,
                    origin=(t * 3, 5, 7), look_at=(0, 0.5, 0), up=(0, 1, 0))

    renderer = Renderer('pt', '../output/frames/frame_%d.png' % i, overwrite=True)
    renderer.initialize(width=width, height=height, min_path_length=1, max_path_length=10,
                        initial_radius=0.05, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=3,
                        num_threads=1)
    renderer.set_camera(camera.c)

    air = VolumeMaterial('vacuum', scattering=0.01)

    scene = Scene()
    with scene:

        scene.set_atmosphere_material(air)

        #scene.add_mesh(create_object('suzanne', 1.5, material='snow'))
        #scene.add_mesh(create_object('suzanne', -1.5, material='snow_nosss'))
        #scene.add_mesh(create_object('cube', 0, material='glossy'))
        #scene.add_mesh(create_object('sphere', -2.5, material='glass'))
        #scene.add_mesh(create_object('holder', 0, -1, -4, 2, material='wall'))
        scene.add_mesh(create_holder('holder', 0, -1, -5, 2, taichi_s=0.7 + 0.25 * t, t=-t*math.pi*2))

        #scene.add_mesh(create_object('plane', 0, -2, 0, 2, material='wall'))
        #create_fractal(scene)
        scene.add_mesh(create_light(math.pi * 0.5))

        #envmap = EnvironmentMap('base', filepath='c:/tmp/20060807_wells6_hd.hdr')

        #scene.set_envmap(envmap)

    renderer.set_scene(scene)
    renderer.render(30000000)

if __name__ == '__main__':
    frames = 120
    for i in [30]:
        render_frame(i, 2 * (-0.5 + 1.0 * i / frames))
    from taichi.tools import video
    video.make_video('../output/frames/%d.png', 960, 540, '../output/video.mp4')
