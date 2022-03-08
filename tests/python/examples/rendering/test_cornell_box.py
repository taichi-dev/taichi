import argparse

import taichi as ti
from tests import test_utils

FRAMES = 200


@test_utils.test(arch=ti.gpu)
def test_cornell_box():
    from taichi.examples.rendering.cornell_box import render, tonemap
    for i in range(FRAMES):
        render()
        interval = 10
        if i % interval == 0:
            tonemap(i)


def video_cornell_box(result_dir):
    from taichi.examples.rendering.cornell_box import (render, tonemap,
                                                       tonemapped_buffer)
    video_manager = ti.VideoManager(output_dir=result_dir,
                                    framerate=24,
                                    automatic_build=False)
    gui = ti.GUI("Taichi Cornell Box",
                 res=800,
                 background_color=0x112F41,
                 show_gui=False)
    for i in range(FRAMES):
        render()
        interval = 10
        if i % interval == 0:
            tonemap(i)

        gui.set_image(tonemapped_buffer)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cornell_box video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_cornell_box(parser.parse_args().output_directory)
