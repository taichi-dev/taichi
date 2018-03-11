import taichi as tc

def create_scene():
  res = 960, 540
  camera = tc.Camera(
      'pinhole',
      res=res,
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
  renderer = tc.Renderer(scene=create_scene())
  renderer.render()
