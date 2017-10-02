import taichi as tc
import random


def create_taichi_text():
  text = 1 - tc.Texture(
      'text',
      content='Taichi',
      width=200,
      height=200,
      font_file=tc.get_asset_path('fonts/go/Go-Bold.ttf'),
      size=50,
      dx=0,
      dy=0)
  mesh = tc.Mesh(
      'plane',
      tc.SurfaceMaterial(
          'transparent',
          nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
          mask=text),
      translate=(5.0, 2, 0.05),
      scale=2,
      rotation=(90, 0, 0))
  return mesh


def create_scene():
  downsample = 1
  width, height = 900 / downsample, 600 / downsample
  camera = tc.Camera(
      'pinhole',
      width=width,
      height=height,
      fov=60,
      origin=(0, 10, 0),
      look_at=(0, 0, 0),
      up=(0, 0, -1))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    text_tex = tc.Texture(
        'image', filename=tc.get_asset_path('textures/graphic_design.png'))

    for i in range(3):
      with tc.transform_scope(
          translate=(0, 0.101, 0), scale=(8, 4, 0.2), rotation=(-90, 0, 0)):
        with tc.transform_scope(scale=1**i):
          mesh = tc.create_volumetric_block(text_tex * 8, res=(512, 256, 4))
          scene.add_mesh(mesh)

    ground_tex = tc.Texture(
        'image', filename=tc.get_asset_path('textures/metal.jpg'))

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('pbr', diffuse_map=ground_tex),
        translate=(0, 0, 0),
        scale=10,
        rotation=(0, 0, 0))
    scene.add_mesh(mesh)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(-10, 3, 5),
        scale=1,
        rotation=(0, 0, -90))
    scene.add_mesh(mesh)

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='volumetric_text', overwrite=True)
  renderer.initialize(
      preset='pt',
      scene=create_scene(),
      min_path_length=1,
      max_path_length=20,
      luminance_clamping=0.1)
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(exposure=0.9, bloom_radius=0.0, gamma=2.2))
  renderer.render(10000, 20)
