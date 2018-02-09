import taichi as tc
import math
import random
import colorsys


def create_scene():
  downsample = 1
  width, height = 960 // downsample, 540 // downsample
  camera = tc.Camera(
      'pinhole',
      width=width,
      height=height,
      fov=90,
      origin=(0, 0, 10),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    for i in range(3):
      with tc.TransformScope(translate=(i, 0, 0)):
        for j in range(3):
          with tc.TransformScope(translate=(0, j, 0)):
            mesh = tc.Mesh(
                'plane',
                tc.SurfaceMaterial('pbr', diffuse=(.1, .1, .1)),
                translate=(0, 0, -0.05),
                scale=0.4,
                rotation=(90.3, 0, 0))
            scene.add_mesh(mesh)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(-30, 30, 10),
        scale=2,
        rotation=(0, 0, -90))

    scene.add_mesh(mesh)

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer('scoping', overwrite=True)

  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(exposure=0.5, bloom_radius=0.0))
  renderer.render(10000, 20)
