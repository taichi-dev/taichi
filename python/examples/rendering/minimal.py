import taichi as tc


def create_scene():
  downsample = 2
  width, height = 800 / downsample, 800 / downsample
  camera = tc.Camera(
      'pinhole',
      res=(width, height),
      fov=40,
      origin=(0, 10, 40),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(0, 0, 0),
        scale=5,
        rotation=(0, 0, 0))
    scene.add_mesh(mesh)

    #scene.add_mesh(tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse=(1, 1, 1)), scale=20,
    #                       translate=(0, 0, 0), rotation=(90, 0, 0)))

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='geometry', overwrite=True)
  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(exposure=2, bloom_radius=0.0))
  renderer.render(10000, 20)
