import taichi as tc
import colorsys


def create_scene():
  downsample = 1
  width, height = 960 // downsample, 540 // downsample
  camera = tc.Camera(
      'pinhole',
      res=(width, height),
      fov=90,
      origin=(0, 0, 10),
      look_at=(0, 0, 0),
      up=(0, 1, 0))

  scene = tc.Scene()

  with scene:
    scene.set_camera(camera)

    taichi_tex = tc.Texture('taichi', scale=0.96)
    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('pbr', diffuse=(.1, .1, .1)),
        translate=(0, 0, -0.05),
        scale=10,
        rotation=(90.3, 0, 0))
    scene.add_mesh(mesh)

    # Taichi Text
    text = 1 - tc.Texture(
        'text',
        content='Taichi',
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
            nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
            mask=text),
        translate=(5.0, 2, 0.05),
        scale=2,
        rotation=(90, 0, 0))
    scene.add_mesh(mesh)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('pbr', diffuse=(0.2, 0.5, 0.2)),
        translate=(0, 0, 0),
        scale=(8.3, 1, 4.5),
        rotation=(90, 0, 0))
    scene.add_mesh(mesh)
    '''
        text = 1 - tc.Texture('text', content='Physically based Computer Graphics', width=400, height=400,
                              size=30,
                              font_file=tc.get_asset_path('fonts/go/Go-Bold.ttf'),
                              dx=0, dy=0)
        mesh = tc.Mesh('plane', tc.SurfaceMaterial('transparent',
                                                   nested=tc.SurfaceMaterial('diffuse', color=(1, 0.1, 0.5)),
                                                   mask=text),
                       translate=(3.0, -6, 0.03), scale=(2, 2, 2), rotation=(90, 0, 0))
        scene.add_mesh(mesh)
        '''

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('pbr', diffuse=(0.2, 0.5, 0.2)),
        translate=(0, 0, 0),
        scale=(8.3, 1, 4.5),
        rotation=(90, 0, 0))
    scene.add_mesh(mesh)

    ring_tex = 1 - tc.Texture('ring', inner=0.0, outer=1.0)
    grid_tex = (1 - tc.Texture('rect', bounds=(0.9, 0.9, 1.0))).repeat(6, 6, 1)

    # Taichi circle
    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial(
            'transparent',
            nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
            mask=taichi_tex),
        translate=(-3.7, 0, 0.05),
        scale=2,
        rotation=(90, 0, 0))
    scene.add_mesh(mesh)

    for i in range(1, 5):
      inv_ring_tex = tc.Texture('ring', inner=0.0, outer=0.5 + i * 0.1)
      color = colorsys.hls_to_rgb(i * 0.1, 0.5, 1.0)
      scene.add_mesh(
          tc.Mesh(
              'plane',
              tc.SurfaceMaterial(
                  'transparent',
                  nested=tc.SurfaceMaterial('diffuse', color=color),
                  mask=inv_ring_tex),
              translate=(-3.7, 0, i * 0.03),
              scale=4,
              rotation=(90, 0, 0)))

    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial(
                'transparent',
                nested=tc.SurfaceMaterial('diffuse', color=(0, 0.2, 0.5)),
                mask=grid_tex),
            translate=(4.3, 0, 0.17),
            scale=1,
            rotation=(90, 0, 0)))

    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial(
                'transparent',
                nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 0)),
                mask=grid_tex),
            translate=(4.3, 0, 0.07),
            scale=2,
            rotation=(90, 0, 0)))

    scene.add_mesh(
        tc.Mesh(
            'plane',
            tc.SurfaceMaterial(
                'transparent',
                nested=tc.SurfaceMaterial('diffuse', color=(0, 1, 1)),
                mask=grid_tex),
            translate=(4.3, 0, 0.02),
            scale=3,
            rotation=(90, 0, 0)))

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(-30, 30, 10),
        scale=6,
        rotation=(0, 0, -90))
    scene.add_mesh(mesh)

    mesh = tc.Mesh(
        'plane',
        tc.SurfaceMaterial('emissive', color=(1, 1, 1)),
        translate=(30, 0, 10),
        scale=2,
        rotation=(0, 0, 90))
    scene.add_mesh(mesh)

  return scene


if __name__ == '__main__':
  renderer = tc.Renderer(output_dir='paper_cut', scene=create_scene())
  renderer.render()
