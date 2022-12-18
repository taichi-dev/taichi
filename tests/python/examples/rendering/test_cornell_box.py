import argparse
import os
from importlib import reload

import pytest

import taichi as ti

FRAMES = 200


@pytest.mark.run_in_serial
@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def test_cornell_box():
    import taichi.examples.rendering.cornell_box as cornell_box
    reload(cornell_box)

    for i in range(FRAMES):
        cornell_box.render()
        interval = 10
        if i % interval == 0:
            cornell_box.tonemap(i)


@pytest.mark.run_in_serial
@pytest.mark.skipif(os.environ.get('TI_LITE_TEST') or '0', reason='Lite test')
def video_cornell_box(result_dir):
    import taichi.examples.rendering.cornell_box as cornell_box
    reload(cornell_box)

    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)
    gui = ti.GUI("Taichi Cornell Box",
                 res=800,
                 background_color=0x112F41,
                 show_gui=False)
    for i in range(FRAMES):
        cornell_box.render()
        interval = 10
        if i % interval == 0:
            cornell_box.tonemap(i)

        gui.set_image(cornell_box.tonemapped_buffer)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cornell_box video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_cornell_box(parser.parse_args().output_directory)
