import numpy as np
from taichi_utils import *
# TODO: Remove cv2
import cv2

class ConfigInitializable:
    def __init__(self, **kwargs):
        self.c = None
        self.c.initialize(config_from_dict(kwargs))

def P(**kwargs):
    return config_from_dict(kwargs)

class Renderer(object):
    def __init__(self, name):
        self.c = tc.create_renderer(name)

    def initialize(self, **kwargs):
        self.c.initialize(config_from_dict(kwargs))

    def render(self, stages):
        for i in range(stages):
            print 'stage', i
            self.render_stage()
            self.show()

    def show(self):
        renderer.write_output('a.png')
        cv2.imshow('Rendered', cv2.imread('a.png'))
        cv2.waitKey(1)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)

class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

def create_object(name, x, y=0, z=0, s=1):
    mesh = tc.create_mesh()
    mesh.initialize(P(filename='../assets/meshes/%s.obj' % name))
    material = tc.create_material('pbr')
    material.initialize(P(diffuse_color=(0.8, 0.8, 1), specular_color=(0.1, 0.1, 0.1), glossiness=-1, transparent=False))
    mesh.set_material(material)
    mesh.translate(Vector(x, y, z))
    mesh.scale_s(s)
    return mesh

def create_light():
    mesh = tc.create_mesh()
    mesh.initialize(P(filename='../assets/meshes/plane.obj'))
    material = tc.create_material('emissive')
    material.initialize(P(color=(20, 20, 20)))
    mesh.set_material(material)
    mesh.translate(Vector(3, 3, -1))
    mesh.scale_s(1)
    mesh.rotate_euler(Vector(0, 0, 135))
    return mesh

if __name__ == '__main__':
    width, height = 960, 540
    camera = Camera('perspective', aspect_ratio=float(width) / height, fov_angle=90,
                    origin=(0, 3, 10), look_at=(0, 0, 0), up=(0, 1, 0))

    renderer = Renderer('pt')
    renderer.initialize(width=960, height=540, min_path_length=1, max_path_length=10,
                        initial_radius=0.05, sampler='sobol')
    renderer.set_camera(camera.c)

    scene = tc.create_scene()

    scene.add_mesh(create_object('cylinder', -6))
    scene.add_mesh(create_object('suzanne', 0))
    scene.add_mesh(create_object('sphere', -3))
    scene.add_mesh(create_object('cone', 3))
    scene.add_mesh(create_object('icosphere', 6))
    scene.add_mesh(create_object('plane', 0, -1, 0, 10))
    scene.add_mesh(create_light())

    scene.finalize()

    renderer.set_scene(scene)
    renderer.render(10000)

    cv2.waitKey(0)
