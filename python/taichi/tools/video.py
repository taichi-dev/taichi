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
        ndarray_to_array2d(img).write(self.directory + '/%05d.png' % self.frame_counter)
        self.frame_counter += 1

    def write_frames(self, images):
        for img in images:
            self.write_frame(img)

    def make_video(self):
        command = "ffmpeg -framerate 24 -i " + self.directory + '/%05d.png' + \
                  " -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + self.directory + '/video.mp4'
        os.system(command)


def make_video(input_files, width=0, height=0, output_path='.'):
    command = "ffmpeg -framerate 24 -i " + input_files + \
              " -s:v " + str(width) + 'x' + str(height) + \
              " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path
    os.system(command)
