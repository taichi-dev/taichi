import argparse
import os

import pytest

import taichi as ti

FRAMES = 100


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_ad_gravity():
    from taichi.examples.simulation.ad_gravity import init, substep

    init()
    for _ in range(FRAMES):
        for _ in range(50):
            substep()


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_ad_gravity(result_dir):
    import numpy as np
    from taichi.examples.simulation.ad_gravity import init, substep, x

    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)

    gui = ti.GUI('Autodiff gravity', show_gui=False)
    init()
    for _ in range(FRAMES):
        for _ in range(50):
            substep()
        gui.circles(x.to_numpy(), radius=3)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ad_gravity video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_ad_gravity(parser.parse_args().output_directory)
