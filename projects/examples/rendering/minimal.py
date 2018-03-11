import taichi as tc


def create_scene():
  camera = tc.Camera(
      'pinhole',
      res=(800, 800),
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

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(scene=create_scene())
  renderer.render()
