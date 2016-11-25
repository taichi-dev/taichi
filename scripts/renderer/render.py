import numpy as np
from taichi_utils import *
# TODO: Remove cv2
import cv2
import math

import renderer.assets as assets

class ConfigInitializable:
    def __init__(self, **kwargs):
        self.c = None
        self.c.initialize(config_from_dict(kwargs))


class Renderer(object):
    def __init__(self, name, output_fn):
        self.c = tc.create_renderer(name)
        self.output_fn = output_fn

    def initialize(self, **kwargs):
        self.c.initialize(config_from_dict(kwargs))

    def render(self, stages):
        for i in range(stages):
            print 'stage', i
            self.render_stage()
            self.show()

    def show(self):
        self.write_output(self.output_fn)
        cv2.imshow('Rendered', cv2.imread(self.output_fn))
        cv2.waitKey(1)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)

class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

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
    material.initialize(P(color=(20, 20, 20)))
    mesh.set_material(material)
    print t
    mesh.translate(Vector(math.cos(t) * -3, 3, -1))
    mesh.scale_s(1)
    mesh.rotate_euler(Vector(0, 0, 180 + math.cos(t) * 45))
    return mesh

def render_frame(i, t):
    width, height = 960, 540
    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=90,
                    origin=(0, 3, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    renderer = Renderer('pt', '../output/frames/%d.png' % i)
    renderer.initialize(width=960, height=540, min_path_length=1, max_path_length=30,
                        initial_radius=0.05, sampler='sobol', russian_roulette=True, volmetric=True, direct_lighting=True)
    renderer.set_camera(camera.c)

    air = tc.create_volume_material("vacuum")
    air.initialize(P(scattering=0.01))

    scene = tc.create_scene()
    scene.set_atmosphere_material(air)

    #scene.add_mesh(create_object('cylinder', -6, material='mirror'))
    #scene.add_mesh(create_object('suzanne', 0, material='interface'))
    scene.add_mesh(create_object('sphere', -3, material='interface'))
    #scene.add_mesh(create_object('cone', 3))
    #scene.add_mesh(create_object('icosphere', 6))
    scene.add_mesh(create_object('plane', 0, -1, 0, 10))
    scene.add_mesh(create_light(t))

    scene.finalize()

    renderer.set_scene(scene)
    renderer.render(100)


if __name__ == '__main__':
    frames = 200
    for i in range(frames):
        print i
        render_frame(i, 2 * math.pi * i / float(frames))
    from tools import video
    video.make_video('../output/frames/%d.png', 960, 540, '../output/video.mp4')
