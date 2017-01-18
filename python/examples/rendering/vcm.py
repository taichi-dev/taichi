import taichi as tc
import random
import colorsys

if __name__ == '__main__':
    downsample = 2
    width = 960 / downsample
    height = 540 / downsample

    scene = tc.Scene()
    with scene:
        camera = tc.Camera('pinhole', width=width, height=height, fov=90,
                           origin=(-5, 2, 0), look_at=(0, 1, 2), up=(0, 1, 0), aperture=10)
        scene.set_camera(camera)

        # Box
        scene.add_mesh(tc.Mesh(
            'cube', material=tc.SurfaceMaterial('pbr',
                                                diffuse=(1, 1, 1)
                                                ), scale=10))

        tex = tc.Texture('checkerboard', tex1=tc.Texture('const', value=(1, 1, 0.5, 1)),
                         tex2=tc.Texture('const', value=(0.5, 0.5, 0.5)),
                         repeat_u=10, repeat_v=10)

        # Ground
        scene.add_mesh(tc.Mesh(
            tc.geometry.create_plane(), material=tc.SurfaceMaterial('pbr',
                                                                    diffuse_map=tex,
                                                                    ), scale=10))

        # Mirror
        scene.add_mesh(tc.Mesh(
            'plane',
            material=tc.SurfaceMaterial('pbr', glossiness=-1, specular=(1, 1, 1)),
            scale=3,
            translate=(0, 0, 2.3),
            rotation=(90, 20, 0)
        ))

        # Spheres
        scene.add_mesh(tc.Mesh(
            tc.geometry.create_sphere((30, 30)),
            material=tc.SurfaceMaterial('pbr', glossiness=0, transparent=True, ior=1.5, specular=(1, 0, 0)),
            scale=1,
            translate=(0, 1.5, -1)
        ))

        scene.add_mesh(tc.Mesh(
            tc.geometry.create_sphere((30, 30)),
            material=tc.SurfaceMaterial('pbr', glossiness=0, specular=(1, 0.1, 0)),
            scale=1,
            translate=(2, 1.5, -1)
        ))

        # Light sources
        scene.add_mesh(tc.Mesh(
            tc.geometry.create_plane(), material=tc.SurfaceMaterial('emissive', color=(0.5, 1, 1)), scale=0.1,
            translate=(0, 8, -9), rotation=(0, 0, 180)
        ))

    renderer = tc.Renderer(preset='vcm', output_dir='sds', scene=scene, max_path_length=10)
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=1, gamma=2.2))

    renderer.render(1000, 20)
