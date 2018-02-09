import taichi as tc
import colorsys


def create_scene():
  downsample = 1
  width, height = 600 // downsample, 900 // downsample
  camera = tc.Camera(
      'pinhole',
      width=width,
      height=height,
      fov=30,
      origin=(0, 12, 20),
      look_at=(0, 0.5, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    ground_tex = tc.Texture(
        'image', filename=tc.get_asset_path('textures/paper.jpg'))

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('pbr', diffuse_map=ground_tex),
        translate=(0, 0, 0),
        scale=10,
        rotation=(0, 0, 0))
    scene.add_mesh(mesh)

    text = 1 - tc.Texture(
        'text',
        content='taichi',
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
            nested=tc.SurfaceMaterial('emissive', color=(10, 10, 10)),
            mask=text),
        translate=(0.20, 3.3, 0),
        scale=0.4,
        rotation=(90, 0, 0))
    scene.add_mesh(mesh)

    with tc.transform_scope(translate=(0, 1, 0), rotation=(0, -20, 0)):
      grid_tex = (1 - tc.Texture('rect', bounds=(0.8, 0.8, 1.0))).repeat(
          5, 5, 1)
      tex = tc.Texture(
          'image', filename=tc.get_asset_path('textures/paper.jpg'))
      material = tc.SurfaceMaterial(
          'transparent',
          nested=tc.SurfaceMaterial('reflective', color_map=tex),
          mask=grid_tex)
      for i in range(1):
        mesh = tc.Mesh(
            'plane',
            material,
            translate=(0, 0.4, -i * 0.3),
            scale=(1, 1, 1.4),
            rotation=(90, 0, 0))
        scene.add_mesh(mesh)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(9, 9, 9)),
        translate=(1, 3, -4),
        scale=0.1,
        rotation=(150, 0, 0))
    scene.add_mesh(mesh)

    emission = 0.001
    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(emission, emission, emission)),
        translate=(0, 10, 0),
        scale=10,
        rotation=(180, 0, 0))
    scene.add_mesh(mesh)

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='gunter_rambow', overwrite=True)
  renderer.initialize(
      preset='pt',
      scene=create_scene(),
      min_path_length=2,
      max_path_length=4,
      luminance_clamping=0.01)
  renderer.set_post_processor(
      tc.post_process.LDRDisplay(
          exposure=1.5, bloom_radius=0.2, gamma=1.5, bloom_threshold=0.3))
  renderer.render(10000, 20)
