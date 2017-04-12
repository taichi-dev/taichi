from taichi.misc.util import ndarray_to_array2d
from taichi.misc.settings import get_output_path
from taichi.visual.post_process import LDRDisplay
import os

FRAME_FN_TEMPLATE = '%05d.png'
FRAME_DIR = 'frames'

# Write the frames to the disk and then make videos (mp4 or gif) if necessary

class VideoManager:
    def __init__(self, output_dir, width=None, height=None, post_processor=None, framerate=24, automatic_build=True):
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
        if self.post_processor:
            img = self.post_processor.process(img)
        if self.width is None:
            self.width = img.shape[0]
            self.height = img.shape[1]
        assert os.path.exists(self.directory)
        fn = FRAME_FN_TEMPLATE % self.frame_counter
        self.frame_fns.append(fn)
        ndarray_to_array2d(img).write(os.path.join(self.frame_directory, fn))
        self.frame_counter += 1
        if self.frame_counter % self.next_video_checkpoint == 0:
            if self.automatic_build:
                self.make_video()
                self.next_video_checkpoint *= 2

    def write_frames(self, images):
        for img in images:
            self.write_frame(img)

    def clean_frames(self):
        for fn in os.listdir(self.frame_directory):
            if fn.endswith('.png') and fn in self.frame_fns:
                os.remove(fn)

    def make_video(self, mp4=True, gif=True):
        command = "ffmpeg -framerate 24 -i " + os.path.join(self.frame_directory, FRAME_FN_TEMPLATE) + \
                  " -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -y " + self.get_output_filename('.mp4')

        os.system(command)

        if gif:
            # Generate the palette
            palette_name = self.get_output_filename('_palette.png')
            command = "ffmpeg -i %s -vf 'fps=%d,scale=320:640:flags=lanczos,palettegen' -y %s" % (
                self.get_output_filename('.mp4'), self.framerate, palette_name)
            print command
            os.system(command)

            # Generate the GIF
            command = "ffmpeg -i %s -i %s -lavfi paletteuse -y %s" % (
                self.get_output_filename('.mp4'), palette_name,
                self.get_output_filename('.gif'))
            print command
            os.system(command)
            os.remove(palette_name)

        if not mp4:
            os.remove(self.get_output_filename('mp4'))


def make_video(input_files, width=0, height=0, framerate=24, output_path='video.mp4'):
    if isinstance(input_files, list):
        from PIL import Image
        with Image.open(input_files[0]) as img:
            width, height = img.size
        import shutil
        tmp_dir = 'tmp_ffmpeg_dir'
        os.mkdir(tmp_dir)
        for i, inp in enumerate(input_files):
            shutil.copy(inp, os.path.join(tmp_dir, '%06d.png' % i))
        command = ("ffmpeg -framerate %d -i " % framerate) + tmp_dir + "/%06d.png" + \
                  " -s:v " + str(width) + 'x' + str(height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path
        os.system(command)
        for i in range(len(input_files)):
            os.remove(os.path.join(tmp_dir, '%06d.png' % i))
        os.rmdir(tmp_dir)
    elif isinstance(input_files, str):
        assert width != 0 and height != 0
        command = ("ffmpeg -framerate %d -i " % framerate) + input_files + \
                  " -s:v " + str(width) + 'x' + str(height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path
        os.system(command)
    else:
        assert 'input_files should be list (of files) or str (of file template, like "%04d.png") instead of ' + \
               str(type(input_files))
