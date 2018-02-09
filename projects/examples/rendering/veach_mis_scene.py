import math

from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture


def create_mis_scene(eye_position):
  num_light_sources = 4
  num_plates = 5
  light_position = Vector(-0.5, 0)
  downsample = 1
  res = 960 // downsample, 540 // downsample
  camera = Camera(
      'pinhole',
      res=res,
      fov=70,
      origin=(0, eye_position.y, eye_position.x),
      look_at=(0, -0.3, 0),
      up=(0, 1, 0))

  scene = Scene()
  with scene:
    scene.set_camera(camera)
    rep = Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95)
    material = SurfaceMaterial('pbr', diffuse_map=rep.id)
    scene.add_mesh(
        Mesh('holder', material=material, translate=(0, -1, -7), scale=2))
    for i in range(num_light_sources):
      radius = 0.002 * 3**i
      e = 0.01 / radius**2
      material = SurfaceMaterial('emissive', color=(e, e, e))
      mesh = Mesh(
          'sphere',
          material,
          translate=(0.2 * (i - (num_light_sources - 1) * 0.5),
                     light_position.y, light_position.x),
          scale=radius)
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
      mesh = Mesh(
          'plane',
          SurfaceMaterial(
              'pbr',
              diffuse=(0.1, 0.1, 0.1),
              specular=(1, 1, 1),
              glossiness=100 * 3**i),
          translate=(0, board_position.y, board_position.x),
          rotation=(90 - angle, 0, 0),
          scale=(0.4, 0.7, 0.05))
      scene.add_mesh(mesh)

      # envmap = EnvironmentMap('base', filepath='d:/assets/schoenbrunn-front_hd.hdr')
      # scene.set_environment_map(envmap)
  return scene


if __name__ == '__main__':
  renderer = Renderer(output_dir='veach_mis', overwrite=True)

  eye_position = Vector(0.9, -0.3)
  renderer.initialize(preset='pt', scene=create_mis_scene(eye_position))
  renderer.set_post_processor(LDRDisplay(exposure=4, bloom_radius=0.00))
  renderer.render(800)
