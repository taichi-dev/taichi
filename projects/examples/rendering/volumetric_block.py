import taichi as tc


def create_scene():
  res = 1280, 720
  camera = tc.Camera(
      'pinhole',
      res=res,
      fov=30,
      origin=(4, 0, 15),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()
  with scene:
    scene.set_camera(camera)

    emission = 100000
    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(emission, emission, emission)),
        translate=(300, 200, 300),
        scale=30,
        rotation=(-90, 0, 0))
    scene.add_mesh(mesh)

    material = tc.SurfaceMaterial(
        'diffuse', color=(0.5, 1, 1), roughness=1.0, f0=1)
    scene.add_mesh(
        tc.Mesh(
            'cube', material=material, translate=(0, 0, -2.0), scale=(1, 1, 1)))

    material = tc.SurfaceMaterial(
        'diffuse', color=(1, 0.5, 1), roughness=1.0, f0=1)
    scene.add_mesh(
        tc.Mesh(
            'cube', material=material, translate=(0, -2.0, 0), scale=(1, 1, 1)))

    material = tc.SurfaceMaterial(
        'diffuse', color=(1, 1, 0.5), roughness=1.0, f0=1)
    scene.add_mesh(
        tc.Mesh(
            'cube', material=material, translate=(-2.0, 0, 0), scale=(1, 1, 1)))

    envmap_texture = tc.Texture(
        'spherical_gradient',
        inside_val=(10, 10, 10, 10),
        outside_val=(1, 1, 1, 0),
        angle=10,
        sharpness=20)
    envmap = tc.EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
    scene.set_environment_map(envmap)

    vol_tex = tc.Texture('sphere', center=(0.5, 0.5, 0.5), radius=0.5)
    for i in range(3):
      with tc.transform_scope(translate=(i, 0, 0)):
        with tc.transform_scope(scale=1**i):
          mesh = tc.create_volumetric_block(vol_tex, res=(32, 32, 32))
          scene.add_mesh(mesh)

  return scene


def render():
  renderer = tc.Renderer(frame=0)
  renderer.initialize(scene=create_scene(), sampler='prand')
  renderer.render()


if __name__ == '__main__':
  render()
