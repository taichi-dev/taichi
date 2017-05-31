import taichi as tc
import random


def create_taichi_text():
    text = 1 - tc.Texture('text', content='Taichi', width=200, height=200,
                          font_file=tc.get_asset_path('fonts/go/Go-Bold.ttf'),
                          size=50,
                          dx=0, dy=0)
    mesh = tc.Mesh('plane', tc.SurfaceMaterial('transparent',
                                               nested=tc.SurfaceMaterial('diffuse', color=(1, 1, 1)),
                                               mask=text),
                   translate=(5.0, 2, 0.05), scale=2, rotation=(90, 0, 0))
    return mesh


def create_scene():
    downsample = 2
    width, height = 1500 / downsample, 600 / downsample
    camera = tc.Camera('pinhole', width=width, height=height, fov=30,
                       origin=(0, 1, 20), look_at=(0, 2, 0), up=(0, 1, 0))

    scene = tc.Scene()

    with scene:
        scene.set_camera(camera)

        ground_tex = tc.Texture('image', filename=tc.get_asset_path('textures/paper.jpg'))

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=ground_tex),
                       translate=(0, 0, -5), scale=10, rotation=(90, 0, 0))
        scene.add_mesh(mesh)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('pbr', diffuse_map=ground_tex),
                       translate=(0, 0, 0), scale=10, rotation=(0, 0, 0))
        scene.add_mesh(mesh)

        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive_spot', color=(1, 1, 1), exponential=3),
                       translate=(0, 0, -1.5), scale=0.1, rotation=(-101, 0, 0))
        scene.add_mesh(mesh)

        fill_light = 0.03
        mesh = tc.Mesh('plane', tc.SurfaceMaterial('emissive', color=(fill_light, fill_light, 3 * fill_light)),
                       translate=(0, 10, 30), scale=1, rotation=(-90, 0, 0))
        scene.add_mesh(mesh)

        emission = 3
        with tc.transform_scope(rotation=(0, 10, 0)):
            mesh = tc.Mesh('plane',
                           tc.SurfaceMaterial('emissive_spot', color=(emission, emission, emission), exponential=100),
                           translate=(10, 2, 1), scale=0.1, rotation=(0, 0, 100))
            scene.add_mesh(mesh)

        for j in range(3):
            for i in range(14):
                with tc.transform_scope(translate=(i - 7, (random.random() - 0.5) * 0.4, j)):
                    with tc.transform_scope(rotation=(0, 0, 10 - j * 10), translate=(0, -j * 0.3 + i * 0.04 - 0.4, 0)):
                        s = random.random() * 0.5 + 0.8
                        r = random.random()
                        if r < 0.5:
                            shape = 'cube'
                        else:
                            shape = tc.geometry.create_cylinder((100, 2), smooth=False)
                        mesh = tc.Mesh(shape, tc.SurfaceMaterial('diffuse', color=(0.3, 0.2, 0.1)),
                                       scale=(0.4 * s, 1 * s, 0.4 * s),
                                       rotation=(-4, -12, 0))
                        scene.add_mesh(mesh)

    return scene


if __name__ == '__main__':
    renderer = tc.Renderer(output_dir='shadows', overwrite=True)
    renderer.initialize(preset='pt', scene=create_scene(), min_path_length=2, max_path_length=4,
                        luminance_clamping=0.1)
    renderer.set_post_processor(
        tc.post_process.LDRDisplay(exposure=0.2, bloom_radius=0.0, gamma=2.2))
    renderer.render(10000, 20)
