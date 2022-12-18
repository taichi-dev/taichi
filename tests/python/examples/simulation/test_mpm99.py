import argparse
import os
from importlib import reload

import pytest

import taichi as ti

FRAMES = 100


@pytest.mark.run_in_serial
@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_mpm99():
    import taichi.examples.simulation.mpm99 as mpm99
    reload(mpm99)

    mpm99.initialize()
    for i in range(FRAMES):
        for s in range(int(2e-3 // mpm99.dt)):
            mpm99.substep()


@pytest.mark.run_in_serial
@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_mpm99(result_dir):
    import taichi.examples.simulation.mpm99 as mpm99
    reload(mpm99)

    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)
    mpm99.initialize()
    gui = ti.GUI("Taichi MLS-MPM-99",
                 res=512,
                 background_color=0x112F41,
                 show_gui=False)
    for i in range(FRAMES):
        for s in range(int(2e-3 // mpm99.dt)):
            mpm99.substep()
        gui.circles(x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEEF0],
                    palette_indices=mpm99.material)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mpm99 video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_mpm99(parser.parse_args().output_directory)
