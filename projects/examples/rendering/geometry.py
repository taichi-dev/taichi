import colorsys

import taichi as tc


def create_scene():
  downsample = 2
  camera = tc.Camera(
      'pinhole',
      res=(800 // downsample, 800 // downsample),
      fov=40,
      origin=(0, 10, 40),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  meshes = [
      tc.geometry.create_torus((30, 30)),
      tc.geometry.create_mobius((100, 30), 1, 0.4),
      tc.geometry.create_mobius((100, 30), 1, 0.4, 3),
      tc.geometry.create_cone((100, 2)),
      tc.geometry.create_cone((3, 2), smooth=False),
      tc.geometry.create_cylinder((10, 2), smooth=True),
      tc.geometry.create_sphere((10, 10), smooth=False)
  ]

  with scene:
    scene.set_camera(camera)
    meshes_per_row = 4
    distance = 3
    for i, m in enumerate(meshes):
      x, y = i % meshes_per_row + 0.5 - meshes_per_row / 2, \
             i / meshes_per_row + 0.5 - meshes_per_row / 2

      color = colorsys.hls_to_rgb(i * 0.1, 0.4, 1.0)
      scene.add_mesh(
          tc.Mesh(
              m,
              tc.SurfaceMaterial(
                  'pbr', diffuse=color, specular=color, glossiness=300),
              translate=(x * distance, y * distance, 2)))

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(30, 30, 60),
        scale=5,
        rotation=(0, 0, 180))
    scene.add_mesh(mesh)

    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial('pbr', diffuse=(1, 1, 1)),
            scale=20,
            translate=(0, 0, 0),
            rotation=(90, 0, 0)))

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='geometry', overwrite=True)
  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(exposure=2, bloom_radius=0.0))
  renderer.render(10000, 20)
