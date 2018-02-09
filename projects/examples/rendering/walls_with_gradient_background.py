import math
import time

from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
from colorsys import hsv_to_rgb
import taichi as tc


def create_scene():
  downsample = 2
  width, height = 1280 // downsample, 720 // downsample
  camera = Camera(
      'pinhole',
      width=width,
      height=height,
      fov=30,
      origin=(0, 0, 10),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = Scene()
  with scene:
    scene.set_camera(camera)
    tex = Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95) * 0.9

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(10000, 10000, 10000)),
        translate=(30, 20, 30),
        scale=3,
        rotation=(0, 0, 180))
    scene.add_mesh(mesh)

    with tc.transform_scope(rotation=(0, 0, 0), scale=0.8):
      material = SurfaceMaterial(
          'diffuse', color=(1, 0.7, 1), roughness_map=tex.id, f0=1)
      scene.add_mesh(
          Mesh(
              'cube',
              material=material,
              translate=(0, -1, 0),
              scale=(2, 0.02, 1)))
      for i in range(7):
        material = SurfaceMaterial(
            'diffuse',
            color=hsv_to_rgb(i * 0.2, 0.5, 1.0),
            roughness_map=tex.id,
            f0=1)
        scene.add_mesh(
            Mesh(
                'cube',
                material=material,
                translate=(2, 0.3 * (i - 3), 0.2),
                scale=(0.01, 0.10, 0.5)))
      material = SurfaceMaterial(
          'diffuse', color=(1, 1, 1), roughness_map=tex.id, f0=1)
      scene.add_mesh(
          Mesh(
              'cube',
              material=material,
              translate=(0, 0, -1),
              scale=(1.9, 0.9, 0.03)))

    envmap_texture = Texture(
        'spherical_gradient',
        inside_val=(10, 10, 10, 10),
        outside_val=(1, 1, 1, 0),
        angle=10,
        sharpness=20)
    envmap = EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
    scene.set_environment_map(envmap)
  return scene


if __name__ == '__main__':
  renderer = Renderer(overwrite=True)
  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(LDRDisplay(exposure=1, bloom_radius=0.01))
  renderer.render(800)
