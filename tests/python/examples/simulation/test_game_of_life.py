import argparse

import taichi as ti

FRAMES = 100


def test_game_of_life():
    from taichi.examples.simulation.game_of_life import init, run

    init()
    for i in range(FRAMES):
        run()


def video_game_of_life(result_dir):
    import numpy as np
    from taichi.examples.simulation.game_of_life import (alive, img_size, init,
                                                         run)

    video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                          framerate=24,
                                          automatic_build=False)

    gui = ti.GUI('Game of Life', (img_size, img_size), show_gui=False)
    gui.fps_limit = 15

    init()
    for i in range(FRAMES):
        run()

        gui.set_image(
            ti.tools.imresize(alive, img_size).astype(np.uint8) * 255)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate game_of_life video')
    parser.add_argument('output_directory',
                        help='output directory of generated video')
    video_game_of_life(parser.parse_args().output_directory)
