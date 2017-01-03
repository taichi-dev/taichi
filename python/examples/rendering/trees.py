import taichi as tc
import random
import colorsys

def create_tree(scene):
    s = random.random() * 0.7 + 0.7
    translate = (random.random() * 0.4, 0, random.random() * 0.4)
    with tc.transform_scope(scale=(0.7, 0.7, 0.7), translate=translate):
        body_material = tc.SurfaceMaterial('pbr',
                                           diffuse=(1, 1, 1)
                                           )
        '''
        scene.add_mesh(tc.Mesh(
            tc.create_sphere((40, 40), True),
            material=body_material,
            scale=1.0, translate=(0, 1.3, 0)))
        '''
        scene.add_mesh(tc.Mesh(
            'cube', material=tc.SurfaceMaterial('diffuse', color=(1, 1, 0.4)),
            scale=(0.1, 0.5, 0.1), translate=(0, 0.5, 0)))
        scene.add_mesh(tc.Mesh(
            tc.create_cone((4, 2), False),
            material=body_material,
            scale=(0.6, 1.0, 0.6), translate=(0, 1.3, 0)))

        #color = (0.46, 0.55, 0.63)




if __name__ == '__main__':
    downsample = 1
    width = 540 / downsample
    height = 960 / downsample

    scene = tc.Scene()
    with scene:
        camera = tc.Camera('pinhole', width=width, height=height, fov=4,
                           origin=(200, 500, 700), look_at=(0, 0, 0), up=(0, 1, 0), aperture=10)
        scene.set_camera(camera)

        # Ground
        scene.add_mesh(tc.Mesh(
            tc.geometry.create_plane(), material=tc.SurfaceMaterial('pbr',
                                                                    diffuse=tc.color255(225, 182, 166),
                                                                    ),
            scale=30))

        # Trees
        for i in range(-7, 7):
            for j in range(-7, 7):
                if -2 < i < 2 and -2 < j < 2:
                    continue
                else:
                    with tc.transform_scope(translate=(i * 4, 0, j * 4)):
                        create_tree(scene)

        with tc.transform_scope(scale=(3, 3, 3)):
            create_tree(scene)

        # Light sources
        scene.add_mesh(tc.Mesh(
            tc.geometry.create_plane(), material=tc.SurfaceMaterial('emissive', color=(0.5, 1, 1)), scale=100,
            translate=(150, 100, -50), rotation=(180, 0, 0)
        ))

        scene.add_mesh(tc.Mesh(
            tc.geometry.create_plane(), material=tc.SurfaceMaterial('emissive', color=(1, 1, 0.5)), scale=30,
            translate=(-50, 100, -50), rotation=(180, 0, 0)
        ))

    renderer = tc.Renderer(preset='pt', scene=scene)
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=4, gamma=1))

    renderer.render(1000, 20)
