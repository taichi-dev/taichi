import taichi as tc
import math
import random
import colorsys

res = [64, 40]
spp = 30
frames = 200
downsample = 2

particles = []


class Particle:
    def __init__(self, i, j):
        self.i, self.j = i, j
        random_vec = lambda: tc.Vector(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
        self.scales = 4
        self.rotation = [random_vec() for k in range(self.scales)]
        self.radius = [random_vec() * (6 ** k - 1) for k in range(self.scales)]

    def create_mesh(self, t):
        t += 3 - 5.0 * self.i / res[0]
        t = 1 / (1 + math.exp(-t))
        material = tc.SurfaceMaterial('microfacet',
                                      color=colorsys.hls_to_rgb(self.i * 1.0 / res[0], 0.5, 1.0),
                                      f0=1, roughness=0.0
                                      )
        transform = tc.core.Matrix4(1.0)
        with tc.transform_scope(translate=(-(res[0] + 1) * 0.5 + self.i, -(res[1] + 1) * 0.5 + self.j, 0)):
            with tc.transform_scope(transform=transform):
                transform = tc.Transform(scale=0.45, rotation=(90, 0, 0))
                for k in range(self.scales):
                    transform.translate(-self.radius[k])
                    transform.rotate(self.rotation[k] * (1 - t) * 100.0 * (0.5 ** k))
                    transform.translate(self.radius[k])

                with tc.transform_scope(transform=transform.get_matrix()):
                    mesh = tc.Mesh('plane', material)
        return mesh


def create_scene(t):
    width, height = 960 / downsample, 540 / downsample
    camera = tc.Camera('thinlens', width=width, height=height, fov=50, aperture=1, focus=(0, 0, 0),
                       origin=(0, 0, 100), look_at=(0, -0.5, 0), up=(0, 1, 0))

    scene = tc.Scene()

    with scene:
        scene.set_camera(camera)

        for p in particles:
            scene.add_mesh(p.create_mesh(t))

        envmap = tc.EnvironmentMap('base', filepath=tc.settings.get_asset_path('/envmaps/schoenbrunn-front_hd.hdr'))
        envmap.set_transform(tc.core.Matrix4(1.0).rotate_euler(tc.Vector(0, -30, 0)))
        scene.set_environment_map(envmap)

    return scene


def render_frame(t):
    renderer = tc.Renderer()
    renderer.initialize(preset='pt', sampler='sobol', scene=create_scene(t))
    renderer.set_post_processor(tc.post_process.LDRDisplay(exposure=1.0, bloom_radius=0.00))
    renderer.render(spp)
    return renderer.get_output()


if __name__ == '__main__':
    for i in range(res[0]):
        for j in range(res[1]):
            particles.append(Particle(i, j))
    video_manager = tc.VideoManager(output_dir='particles')
    images = []
    for i in range(frames + 1):
        print 'frame', i
        # images.append(render_frame(0.5 * (1 - math.cos(math.pi * i / frames))))
        images.append(render_frame(12.0 * i / frames))

    video_manager.write_frames(images)
    video_manager.make_video()
