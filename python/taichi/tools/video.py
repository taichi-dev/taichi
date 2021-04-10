import os

from taichi.core.settings import get_os_name
from taichi.misc.image import imwrite

FRAME_FN_TEMPLATE = '%06d.png'
FRAME_DIR = 'frames'

# Write the frames to the disk and then make videos (mp4 or gif) if necessary


def scale_video(input, output, ratiow, ratioh):
    os.system('ffmpeg -i {}  -vf "scale=iw*{:.4f}:ih*{:.4f}" {}'.format(
        input, ratiow, ratioh, output))


def crop_video(input, output, x_begin, x_end, y_begin, y_end):
    os.system(
        'ffmpeg -i {} -filter:v "crop=iw*{:.4f}:ih*{:.4f}:iw*{:0.4f}:ih*{:0.4f}" {}'
        .format(input, x_end - x_begin, y_end - y_begin, x_begin, 1 - y_end,
                output))


def accelerate_video(input, output, speed):
    os.system('ffmpeg -i {} -filter:v "setpts={:.4f}*PTS" {}'.format(
        input, 1 / speed, output))


def get_ffmpeg_path():
    return 'ffmpeg'


def mp4_to_gif(input_fn, output_fn, framerate):
    # Generate the palette
    palette_name = 'palette.png'
    if get_os_name() == 'win':
        command = get_ffmpeg_path(
        ) + " -loglevel panic -i %s -vf 'palettegen' -y %s" % (input_fn,
                                                               palette_name)
    else:
        command = get_ffmpeg_path(
        ) + " -loglevel panic -i %s -vf 'fps=%d,scale=320:640:flags=lanczos,palettegen' -y %s" % (
            input_fn, framerate, palette_name)
    # print command
    os.system(command)

    # Generate the GIF
    command = get_ffmpeg_path(
    ) + " -loglevel panic -i %s -i %s -lavfi paletteuse -y %s" % (
        input_fn, palette_name, output_fn)
    # print command
    os.system(command)
    os.remove(palette_name)


class VideoManager:
    def __init__(self,
                 output_dir,
                 width=None,
                 height=None,
                 post_processor=None,
                 framerate=24,
                 automatic_build=True):
        assert (width is None) == (height is None)
        self.width = width
        self.height = height
        self.directory = output_dir
        self.frame_directory = os.path.join(self.directory, FRAME_DIR)
        try:
            os.makedirs(self.frame_directory)
        except:
            pass
        self.next_video_checkpoint = 4
        self.framerate = framerate
        self.post_processor = post_processor
        self.frame_counter = 0
        self.frame_fns = []
        self.automatic_build = automatic_build

    def get_output_filename(self, suffix):
        return os.path.join(self.directory, 'video' + suffix)

    def write_frame(self, img):
        if img.shape[0] % 2 != 0:
            print('Warning: height is not divisible by 2! Dropping last row')
            img = img[:-1]
        if img.shape[1] % 2 != 0:
            print('Warning: width is not divisible by 2! Dropping last column')
            img = img[:, :-1]
        if self.post_processor:
            img = self.post_processor.process(img)
        if self.width is None:
            self.width = img.shape[0]
            self.height = img.shape[1]
        assert os.path.exists(self.directory)
        fn = FRAME_FN_TEMPLATE % self.frame_counter
        self.frame_fns.append(fn)
        imwrite(img, os.path.join(self.frame_directory, fn))
        self.frame_counter += 1
        if self.frame_counter % self.next_video_checkpoint == 0:
            if self.automatic_build:
                self.make_video()
                self.next_video_checkpoint *= 2

    def get_frame_directory(self):
        return self.frame_directory

    def write_frames(self, images):
        for img in images:
            self.write_frame(img)

    def clean_frames(self):
        for fn in os.listdir(self.frame_directory):
            if fn.endswith('.png') and fn in self.frame_fns:
                os.remove(fn)

    def make_video(self, mp4=True, gif=True):
        fn = self.get_output_filename('.mp4')
        command = (get_ffmpeg_path() + " -loglevel panic -framerate %d -i " % self.framerate) + os.path.join(self.frame_directory, FRAME_FN_TEMPLATE) + \
                  " -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p -y " + fn

        os.system(command)

        if gif:
            mp4_to_gif(self.get_output_filename('.mp4'),
                       self.get_output_filename('.gif'), self.framerate)

        if not mp4:
            os.remove(fn)


def interpolate_frames(frame_dir, mul=4):
    # TODO: remove dependency on cv2 here
    import cv2
    files = os.listdir(frame_dir)
    images = []
    images_interpolated = []
    for f in sorted(files):
        if f.endswith('png'):
            images.append(cv2.imread(f) / 255.0)

    for i in range(len(images) - 1):
        images_interpolated.append(images[i])
        for j in range(mul - 1):
            alpha = 1 - j / mul
            images_interpolated.append(images[i] * alpha + images[i + 1] *
                                       (1 - alpha))

    images_interpolated.append(images[-1])

    os.makedirs('interpolated', exist_ok=True)
    for i, img in enumerate(images_interpolated):
        cv2.imwrite('interpolated/{:05d}.png'.format(i), img * 255.0)


def ffmpeg_common_args(frame_rate, input, width, height, crf, output_path):
    return f"{get_ffmpeg_path()} -y -loglevel panic -framerate {frame_rate} -i {input} -s:v {width}x{height} " + \
                         f"-c:v libx264 -profile:v high -crf {crf} -pix_fmt yuv420p {output_path}"


def make_video(input_files,
               width=0,
               height=0,
               frame_rate=24,
               crf=20,
               output_path='video.mp4'):
    if isinstance(input_files, list):
        from PIL import Image
        with Image.open(input_files[0]) as img:
            width, height = img.size
        import shutil
        tmp_dir = 'tmp_ffmpeg_dir'
        os.mkdir(tmp_dir)
        if width % 2 != 0:
            print("Width ({}) not divisible by 2".format(width))
            width -= 1
        if height % 2 != 0:
            print("Height ({}) not divisible by 2".format(width))
            height -= 1
        for i, inp in enumerate(input_files):
            shutil.copy(inp, os.path.join(tmp_dir, '%06d.png' % i))
        inputs = f'{tmp_dir}/%06d.png'
        command = ffmpeg_common_args(frame_rate, inputs, width, height, crf,
                                     output_path)
        ret = os.system(command)
        assert ret == 0, "ffmpeg failed to generate video file."
        for i in range(len(input_files)):
            os.remove(os.path.join(tmp_dir, '%06d.png' % i))
        os.rmdir(tmp_dir)
    elif isinstance(input_files, str):
        assert width != 0 and height != 0
        command = ffmpeg_common_args(frame_rate, input_files, width, height,
                                     crf, output_path)
        ret = os.system(command)
        assert ret == 0, "ffmpeg failed to generate video file."
    else:
        assert False, f'input_files should be list (of files) or str (of file template, e.g., "%04d.png") instead of {type(input_files)}'
