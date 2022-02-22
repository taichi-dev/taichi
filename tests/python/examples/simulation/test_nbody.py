import argparse

import taichi as ti

FRAMES = 100

def test_nbody():
    from taichi.examples.simulation.nbody import (substepping, initialize, compute_force, update)

    initialize()
    for i in range(FRAMES):
        for i in range(substepping):
                compute_force()
                update()


def video_nbody(result_dir):
    from taichi.examples.simulation.nbody import (substepping, pos, planet_radius, initialize, compute_force, update)

    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                    framerate=24,
                                    automatic_build=False)

    initialize()
    gui = ti.GUI('N-body problem', (800, 800), show_gui=False)
    for i in range(FRAMES):
        for i in range(substepping):
            compute_force()
            update()

        gui.circles(pos.to_numpy(), color=0xffffff, radius=planet_radius)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nbody video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_nbody(parser.parse_args().output_directory)
