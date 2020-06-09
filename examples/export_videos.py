import taichi as ti

ti.init()

pixels = ti.var(ti.u8, shape=(512, 512, 3))


@ti.kernel
def paint():
    for i, j, k in pixels:
        pixels[i, j, k] = ti.random() * 255


result_dir = "./results"
video_manger = ti.VideoManager(output_dir=result_dir,
                               framerate=24,
                               automatic_build=False)

for i in range(50):
    paint()

    pixels_img = pixels.to_numpy()
    video_manger.write_frame(pixels_img)
    print(f'\rFrame {i+1}/50 is recorded', end='')

print()
print('Exporting .mp4 and .gif videos...')
video_manger.make_video(gif=True, mp4=True)
print(f'MP4 video is saved to {video_manger.get_output_filename(".mp4")}')
print(f'GIF video is saved to {video_manger.get_output_filename(".gif")}')
