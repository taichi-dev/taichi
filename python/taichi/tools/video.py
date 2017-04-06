from taichi.misc.util import ndarray_to_array2d
from taichi.misc.settings import get_output_path
from taichi.visual.post_process import LDRDisplay
import os


class VideoManager:
    def __init__(self, output_dir, width=None, height=None, post_processor=None):
        assert (width is None) == (height is None)
        self.width = width
        self.height = height
        self.directory = get_output_path(output_dir)
        try:
            os.mkdir(self.directory)
        except:
            pass
        self.post_processor = post_processor
        self.frame_counter = 0

    def write_frame(self, img):
        if self.post_processor:
            img = self.post_processor.process(img)
        if self.width is None:
            self.width = img.shape[0]
            self.height = img.shape[1]
        assert os.path.exists(self.directory)
        ndarray_to_array2d(img).write(os.path.join(self.directory, '%05d.png' % self.frame_counter))
        self.frame_counter += 1

    def write_frames(self, images):
        for img in images:
            self.write_frame(img)

    def make_video(self):
        command = "ffmpeg -framerate 24 -i " + self.directory + '/%05d.png' + \
                  " -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + self.directory + '/video.mp4'
        os.system(command)


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
