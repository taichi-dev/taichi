import sys

FRAMES = 100


def test_mpm99():
    from taichi.examples.simulation.mpm99 import dt, initialize, substep

    initialize()
    for i in range(FRAMES):
        for s in range(int(2e-3 // dt)):
            substep()


def video_mpm99(result_dir):
    from taichi.examples.simulation.mpm99 import (dt, initialize, material,
                                                  substep, ti, x)

    video_manager = ti.VideoManager(output_dir=result_dir,
                                    framerate=24,
                                    automatic_build=False)
    initialize()
    gui = ti.GUI("Taichi MLS-MPM-99",
                 res=512,
                 background_color=0x112F41,
                 show_gui=False)
    for i in range(FRAMES):
        for s in range(int(2e-3 // dt)):
            substep()
        gui.circles(x.to_numpy(),
                    radius=1.5,
                    palette=[0x068587, 0xED553B, 0xEEEEF0],
                    palette_indices=material)
        video_manager.write_frame(gui.get_image())
        gui.clear()
    video_manager.make_video(mp4=True, gif=False)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    video_mpm99(sys.argv[1])
