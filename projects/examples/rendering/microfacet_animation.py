import taichi as tc
import math
import random
import colorsys


def create_scene(t):
  downsample = 1
  width, height = 960 // downsample, 540 // downsample
  camera = tc.Camera(
      'pinhole',
      width=width,
      height=height,
      fov=50,
      origin=(0, 2, 10),
      look_at=(0, -0.5, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  roughness = t * t

  with scene:
    scene.set_camera(camera)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial(
            'microfacet', color=(1, 1, 1), roughness=roughness, f0=1),
        translate=(0, 0, 0),
        scale=7,
        rotation=(0, 0, 0))
    scene.add_mesh(mesh)

    for i in range(5):
      with tc.transform_scope(translate=(1.4 * (i - 2), 0.6, 0)):
        with tc.transform_scope(scale=(0.3, 1, 0.5), rotation=(90, 0, 0)):
          mesh = tc.Mesh('plane',
                         tc.SurfaceMaterial(
                             'emissive',
                             color=colorsys.hls_to_rgb(i * 0.2, 0.5, 1.0)))
          scene.add_mesh(mesh)

  return scene


if __name__ == '__main__':
  frames = 40
  video_manager = tc.VideoManager(output_dir='microfacet_anim')
  images = []
  for i in range(frames + 1):
    renderer = tc.Renderer()
    renderer.initialize(
        preset='pt',
        scene=create_scene(0.5 * (1 - math.cos(math.pi * i / frames))))
    renderer.set_post_processor(
        tc.post_process.LDRDisplay(exposure=1.0, bloom_radius=0.00))
    renderer.render(200)
    images.append(renderer.get_output())

  images = images + images[1:-1][::-1]

  video_manager.write_frames(images)
  video_manager.make_video()
