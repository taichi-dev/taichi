import argparse
import os

import pytest

import taichi as ti

FRAMES = 100


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_nbody():
    from taichi.examples.simulation.nbody import (compute_force, initialize,
                                                  substepping, update)

    initialize()
    for i in range(FRAMES):
        for i in range(substepping):
            compute_force()
            update()


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_nbody(result_dir):
    from taichi.examples.simulation.nbody import (compute_force, initialize,
                                                  planet_radius, pos,
                                                  substepping, update)

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
