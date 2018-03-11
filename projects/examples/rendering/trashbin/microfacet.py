import taichi as tc
import math
import random
import colorsys


def create_scene():
  downsample = 1
  res = (960 // downsample, 540 // downsample)
  camera = tc.Camera(
      'pinhole',
      res=res, fov=50, origin=(0, 2, 5), look_at=(0, 0, 0), up=(0, 1, 0))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    num_spheres = 6
    for i in range(num_spheres):
      with tc.transform_scope(translate=(0.7 * (i - (num_spheres - 1) / 2.0), 0,
                                         0)):
        r = 1.0 * i / (num_spheres - 1)
        r = r * r
        scene.add_mesh(
            tc.Mesh(
                'sphere',
                tc.SurfaceMaterial(
                    'microfacet',
                    color=(1, 1, 0.5),
                    roughness=(0.01 + r, 0, 0, 0),
                    f0=1),
                scale=0.3))

      mesh = tc.Mesh(
          'holder',
          tc.SurfaceMaterial(
              'pbr', diffuse_map=tc.Texture.create_taichi_wallpaper(20)),
          translate=(0, -1, -3),
          scale=1,
          rotation=(0, 0, 0))
      scene.add_mesh(mesh)

    fp = tc.settings.get_asset_path('envmaps/schoenbrunn-front_hd.hdr')
    envmap = tc.EnvironmentMap('base', filepath=fp)
    envmap.set_transform(
        tc.core.Matrix4(1.0).rotate_euler(tc.Vector(0, -30, 0)))
    scene.set_environment_map(envmap)

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='microfacet', overwrite=True)

  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(exposure=1.0, bloom_radius=0.05))
  renderer.render(10000, 10)
