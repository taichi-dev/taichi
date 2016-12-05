from visual.renderer import *
from visual.texture import Texture
import json

def map_filename(name):
    if name.rfind('/') == -1:
        filename = '../assets/meshes/%s.obj' % name
    else:
        filename = name
    return filename


def create_object(name, x, y=0, z=0, s=1, r=(0, 0, 0), material='wall'):
    mesh = tc.create_mesh()
    mesh.initialize(P(filename=map_filename(name)))
    mesh.set_material(assets.materials.get_material(material))
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    mesh.rotate_euler(Vector(*r))
    return mesh

def create_light():
    mesh = tc.create_mesh()
    mesh.initialize(P(filename='../assets/meshes/plane.obj'))
    material = tc.create_surface_material('emissive')
    e = 1
    material.initialize(P(color=(e, e, e)))
    mesh.set_material(material)
    mesh.translate(Vector(0, 5, -1))
    mesh.scale_s(1e-4)
    mesh.rotate_euler(Vector(0, 0, 180))
    return mesh

def load_scene(root, fov):
    ROOT = root
    f = json.load(open(ROOT + 'scene.json'))
    bsdfs = f['bsdfs']
    materials = {}
    for bsdf in bsdfs:
        name = bsdf['name']
        print name
        material = tc.create_surface_material('diffuse')
        params = {}

        albedo = bsdf['albedo']
        if isinstance(albedo, float):
            params['diffuse'] = (albedo, albedo, albedo)
        elif isinstance(albedo, list):
            params['diffuse'] = tuple(albedo)
        else:
            tex = Texture('image', filename=ROOT + albedo)
            params['diffuse_map'] = tex.id

        material.initialize(P(**params))
        materials[name] = material

    meshes = []

    for mesh_node in f['primitives']:
        if 'file' in mesh_node:
            # Object
            mesh = tc.create_mesh()
            fn = ROOT + mesh_node['file'][:-4] + '.obj'
            mesh.initialize(P(filename=fn))
            mesh.set_material(materials[mesh_node['bsdf']])
        else:
            # Light source
            mesh = tc.create_mesh()
            mesh.initialize(P(filename='../assets/meshes/plane.obj'))
            material = tc.create_surface_material('emissive')
            e = 1
            material.initialize(P(color=(e, e, e)))
            mesh.set_material(material)
            if 'transform' in mesh_node:
                trans = mesh_node['transform']
                if 'position' in trans:
                    mesh.translate(Vector(*trans['position']))
                if 'rotation' in trans:
                    mesh.rotate_euler(Vector(*trans['rotation']))
        meshes.append(mesh)

    camera_node = f['camera']
    width, height = camera_node['resolution']
    # the FOV value is ?
    #fov = math.degrees(math.atan(27.2 / camera_node['fov']) * 2)

    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=fov,
                    origin=tuple(camera_node['transform']['position']),
                    look_at=tuple(camera_node['transform']['look_at']),
                    up=tuple(camera_node['transform']['up']))

    return (width, height), meshes, camera

def render():
    downsample = 1
    root, fov = 'D:/assets/living-room-3/', 55
    #root, fov = 'D:/assets/staircase/', 105
    (width, height), meshes, camera = load_scene(root, fov)

    renderer = Renderer('pt', output_dir='../output/frames/%s' % get_uuid())
    renderer.initialize(width=width, height=height, min_path_length=1, max_path_length=10,
                        initial_radius=0.05, sampler='sobol', russian_roulette=False, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=1,
                        num_threads=8)

    air = tc.create_volume_material("vacuum")
    air.initialize(P(scattering=0.00))

    scene = tc.create_scene()
    scene.set_atmosphere_material(air)

    for mesh in meshes:
        scene.add_mesh(mesh)

    scene.finalize()

    renderer.set_camera(camera.c)

    renderer.set_scene(scene)
    renderer.render(30000000, cache_interval=10)

if __name__ == '__main__':
    render()
