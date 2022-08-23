import argparse
import os

import pytest

import taichi as ti

FRAMES = 100


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_taichi_logo():
    from taichi.examples.rendering.taichi_logo import paint
    paint()


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_taichi_logo(result_dir):
    from taichi.examples.rendering.taichi_logo import n, paint, x
    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)
    paint()
    gui = ti.GUI('Logo', (n, n), show_gui=False)
    for i in range(FRAMES):
        gui.set_image(x)
        video_manager.write_frame(gui.get_image())
        gui.clear()

    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate taichi_logo video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_taichi_logo(parser.parse_args().output_directory)
