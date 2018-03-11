import taichi as tc
import colorsys

def create_scene():
  camera = tc.Camera(
      'thinlens',
      res=(960, 540),
      fov=55,
      origin=(30, 40, 50),
      look_at=(0, 0, 0),
      up=(0, 1, 0),
      focus=(0, 3, 0),
      aperture=0.1)

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    # Plane
    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial('pbr', diffuse=(1, 1, 1.0)),
            translate=(0, 0, 0),
            scale=100,
            rotation=(0, 0, 0)))

    for i in range(8):
      menger = 1 - tc.Texture("menger", limit=i)
      scene.add_mesh(
          tc.Mesh(
              'plane',
              tc.SurfaceMaterial(
                  'transparent',
                  mask=menger,
                  nested=tc.SurfaceMaterial(
                      'diffuse',
                      color=colorsys.hls_to_rgb(i * 0.1 + 0.3, 0.3, 1.0))),
              translate=(i * 7 - 28, 3.5, -5),
              scale=3,
              rotation=(90, 0, 0)))

    # Lights
    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
            translate=(0, 100, -200),
            scale=5,
            rotation=(180, 0, 0)))

    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
            translate=(0, 100, 200),
            scale=3,
            rotation=(180, 0, 0)))

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='fractals', overwrite=True)
  renderer.initialize(preset='pt', scene=create_scene())
  renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=1.0, bloom_radius=0.1))
  renderer.render(10000, 20)
