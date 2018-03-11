import taichi as tc
import colorsys


def create_scene():
  camera = tc.Camera(
      'pinhole',
      res=(1280, 720),
      fov=30,
      origin=(0, 0, 10),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()
  with scene:
    scene.set_camera(camera)
    tex = tc.Texture.create_taichi_wallpaper(20, rotation=0, scale=0.95) * 0.9

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(10000, 10000, 10000)),
        translate=(30, 20, 30),
        scale=3,
        rotation=(0, 0, 180))
    scene.add_mesh(mesh)

    with tc.transform_scope(rotation=(0, 0, 0), scale=0.8):
      material = tc.SurfaceMaterial(
          'diffuse', color=(1, 0.7, 1), roughness_map=tex.id, f0=1)
      scene.add_mesh(
          tc.Mesh(
              'cube',
              material=material,
              translate=(0, -1, 0),
              scale=(2, 0.02, 1)))
      for i in range(7):
        material = tc.SurfaceMaterial(
            'diffuse',
            color=colorsys.hsv_to_rgb(i * 0.2, 0.5, 1.0),
            roughness_map=tex.id,
            f0=1)
        scene.add_mesh(
            tc.Mesh(
                'cube',
                material=material,
                translate=(2, 0.3 * (i - 3), 0.2),
                scale=(0.01, 0.10, 0.5)))
      material = tc.SurfaceMaterial(
          'diffuse', color=(1, 1, 1), roughness_map=tex.id, f0=1)
      scene.add_mesh(
          tc.Mesh(
              'cube',
              material=material,
              translate=(0, 0, -1),
              scale=(1.9, 0.9, 0.03)))

    envmap_texture = tc.Texture(
        'spherical_gradient',
        inside_val=(10, 10, 10, 10),
        outside_val=(1, 1, 1, 0),
        angle=10,
        sharpness=20)
    envmap = tc.EnvironmentMap('base', texture=envmap_texture.id, res=(1024, 1024))
    scene.set_environment_map(envmap)
  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(scene=create_scene())
  renderer.render()
