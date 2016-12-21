from taichi.visual import *
from taichi.visual.post_process import LDRDisplay
from taichi.util import *
import json

def load_scene(root, fov):
    scene = Scene()
    ROOT = root
    f = json.load(open(ROOT + 'scene.json'))
    bsdfs = f['bsdfs']
    materials = {}
    for bsdf in bsdfs:
        name = bsdf['name']
        params = {}

        albedo = bsdf['albedo']
        if isinstance(albedo, float):
            params['diffuse'] = (albedo, albedo, albedo)
        elif isinstance(albedo, list):
            params['diffuse'] = tuple(albedo)
        else:
            tex = Texture('image', filename=ROOT + albedo)
            params['diffuse_map'] = tex.id

        material = SurfaceMaterial('pbr', **params)
        materials[name] = material

    meshes = []

    for mesh_node in f['primitives']:
        if 'file' in mesh_node:
            # Object
            fn = ROOT + mesh_node['file'][:-4] + '.obj'
            mesh = Mesh(filename=fn, material=materials[mesh_node['bsdf']])
        else:
            # Light source
            e = 1
            material = SurfaceMaterial('emissive', color=(e, e, e))
            mesh = Mesh(filename='../assets/meshes/plane.obj', material=material)
            if 'transform' in mesh_node:
                trans = mesh_node['transform']
                if 'rotation' in trans:
                    mesh.rotate_euler(Vector(*trans['rotation']))
                if 'position' in trans:
                    mesh.translate(Vector(*trans['position']))
        meshes.append(mesh)

    camera_node = f['camera']
    width, height = camera_node['resolution']
    # the FOV value is ?
    #fov = math.degrees(math.atan(27.2 / camera_node['fov']) * 2)

    camera = Camera('pinhole', width=width, height=height, aspect_ratio=float(width) / height, fov=fov,
                    origin=tuple(camera_node['transform']['position']),
                    look_at=tuple(camera_node['transform']['look_at']),
                    up=tuple(camera_node['transform']['up']))
    with scene:
        scene.set_camera(camera)
        for mesh in meshes:
            scene.add_mesh(mesh)

    return scene

def render():
    #root, fov = '../../taichi_assets/scenes/living-room/', 55
    root, fov = '../../taichi_assets/scenes/staircase/', 105
    scene = load_scene(root, fov)

    renderer = Renderer('pt', output_dir='../output/frames/reader', overwrite=True)
    renderer.set_post_processor(LDRDisplay(1.0))

    renderer.set_scene(scene)
    renderer.initialize(min_path_length=1, max_path_length=10,
                        initial_radius=0.05, sampler='sobol', russian_roulette=True, volmetric=True, direct_lighting=1,
                        direct_lighting_light=1, direct_lighting_bsdf=1, envmap_is=1, mutation_strength=1, stage_frequency=1,
                        num_threads=8)
    renderer.render(30000, cache_interval=10)

if __name__ == '__main__':
    render()
