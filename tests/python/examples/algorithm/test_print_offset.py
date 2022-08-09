import argparse
import os

import pytest
from taichi.lang import impl

import taichi as ti

FRAMES = 100


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_print_offset():
    from taichi.examples.algorithm.print_offset import fill
    fill()


@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_print_offset(result_dir):
    from taichi.examples.algorithm.print_offset import a, fill, m, n
    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)

    fill()

    gui = ti.GUI('layout',
                 res=(256, 512),
                 background_color=0xFFFFFF,
                 show_gui=False)

    for f in range(FRAMES):
        for i in range(1, m):
            gui.line(begin=(0, i / m),
                     end=(1, i / m),
                     radius=2,
                     color=0x000000)
        for i in range(1, n):
            gui.line(begin=(i / n, 0),
                     end=(i / n, 1),
                     radius=2,
                     color=0x000000)
        for i in range(n):
            for j in range(m):
                gui.text(f'{a[i, j]}', ((i + 0.3) / n, (j + 0.75) / m),
                         font_size=30,
                         color=0x0)
        video_manager.write_frame(gui.get_image())
        gui.clear()

    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate print_offset video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_print_offset(parser.parse_args().output_directory)
