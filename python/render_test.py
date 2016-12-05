from taichi.visual import *
from taichi.util import Vector
from taichi.visual.texture import Texture
from taichi.visual import assets
import math


def map_filename(name):
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = name
    return filename

def create_object(name, x, y=0, z=0, s=1, r=(0, 0, 0), material='wall'):
    mesh = Mesh(map_filename(name), material=assets.materials.get_material(material))
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    mesh.rotate_euler(Vector(*r))
    return mesh

def create_holder(name, x, y=0, z=0, s=1, r=(0, 0, 0), t=0, taichi_s=0.9):
    rep = Texture.create_taichi_wallpaper(20, rotation=t, scale=taichi_s)
    diff = rep
    material = SurfaceMaterial('pbr', diffuse_map=diff.id, glossiness=-1)

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
            mesh = create_object('cube', (i - (n - 1) / 2.0) * s, j * s - 0.5, 0, s * 0.3, r=(i * 10, j * 10, 0),
                                 material='glass')
            scene.add_mesh(mesh)

def create_light(t):
    e = 100
    material = SurfaceMaterial('emissive', color=(e, e, e))
    mesh = Mesh('../assets/meshes/sphere.obj', material)
    mesh.translate(Vector(math.cos(t) * -3, 4, -1))
    mesh.scale_s(0.5)
    mesh.rotate_euler(Vector(0, 0, 180 + math.cos(t) * 45))
    return mesh

def create_mis_scene(eye_position):
    scene = Scene()
    num_light_sources = 4
    num_plates = 5
    light_position = Vector(-0.5, 0)
    with scene:
        e = 1
        material = SurfaceMaterial('emissive', color=(e, e, e))
        mesh = Mesh('../assets/meshes/plane.obj', material)
        mesh.translate(Vector(2, 1, 0))
        mesh.scale_s(1)
        mesh.rotate_euler(Vector(0, 0, 90))
        scene.add_mesh(mesh)
        #scene.add_mesh(create_object('plane', 0, -0.6, 0, 2, r=(0, 180, 0)))
        scene.add_mesh(create_holder('holder', 0, -1, -7, 2))
        for i in range(num_light_sources):
            radius = 0.002 * 3 ** i
            e = 0.01 / radius**2
            material = SurfaceMaterial('emissive', color=(e, e, e))
            mesh = Mesh('../assets/meshes/sphere.obj', material)
            mesh.translate(Vector(0.2 * (i - (num_light_sources - 1) * 0.5), light_position.y, light_position.x))
            mesh.scale_s(radius)
            scene.add_mesh(mesh)

        for i in range(num_plates):
            fraction = -math.pi / 2 - 1.0 * i / num_plates * 0.9
            z = math.cos(fraction) * 1
            y = math.sin(fraction) * 1 + 0.5
            board_position = Vector(z, y)
            vec1 = eye_position - board_position
            vec2 = light_position - board_position
            vec1 *= 1.0 / math.hypot(vec1.x, vec1.y)
            vec2 *= 1.0 / math.hypot(vec2.x, vec2.y)
            half_vector = vec1 + vec2
            angle = math.degrees(math.atan2(half_vector.y, half_vector.x))
            print angle
            mesh = Mesh('../assets/meshes/plane.obj', SurfaceMaterial('pbr', diffuse=(0.1, 0.1, 0.1), specular=(1, 1, 1), glossiness=100 * 3 ** i))
            #mesh = Mesh('../assets/meshes/plane.obj', SurfaceMaterial('diffuse', diffuse=(1, 1, 1)))
            mesh.translate(Vector(0, board_position.y, board_position.x))
            mesh.rotate_euler(Vector(90-angle, 0, 0))
            mesh.scale(Vector(0.4, 0.7, 0.05))
            scene.add_mesh(mesh)

        #envmap = EnvironmentMap('base', filepath='d:/assets/schoenbrunn-front_hd.hdr')
        #scene.set_environment_map(envmap)
    return scene

def render_frame(i, t):
    downsample = 2
    width, height = 960 / downsample, 540 / downsample
    eye_position = Vector(0.9, -0.3)
    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=70,
                    origin=(0, eye_position.y, eye_position.x), look_at=(0, -0.3, 0), up=(0, 1, 0))

    renderer = Renderer('pt', '../output/frames/frame_%d.png' % i, overwrite=True)
    renderer.initialize(width=width, height=height, min_path_length=1, max_path_length=2,
                        initial_radius=0.05, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=3,
                        num_threads=1)
    renderer.set_camera(camera.c)
    renderer.set_scene(create_mis_scene(eye_position))
    renderer.render(30000000)

if __name__ == '__main__':
    frames = 120
    for i in [30]:
        render_frame(i, 2 * (-0.5 + 1.0 * i / frames))
    from taichi.tools import video
    video.make_video('../output/frames/%d.png', 960, 540, '../output/video.mp4')
