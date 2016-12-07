from taichi.visual.post_process import LDRDisplay
import os
import cv2

class VideoManager:
    def __init__(self, directory, width, height):
        self.width = width
        self.height = height
        self.directory = '../output/frames/' + directory
        try:
            os.mkdir(self.directory)
        except Exception as e:
            print e
        self.frame_counter = 0

    def write_frame(self, img):
        img = LDRDisplay().process(img) * 255.0
        cv2.imwrite(self.directory + '/%05d.png' % self.frame_counter, img)
        self.frame_counter += 1

    def make_video(self):
        command = "ffmpeg -framerate 24 -i " + self.directory + '/%05d.png' + \
                  " -s:v " + str(self.width) + 'x' + str(self.height) + \
                  " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + self.directory + '/video.mp4'
        os.system(command)


def make_video(input_files, width, height, output_path):
    command = "ffmpeg -framerate 24 -i " + input_files + \
              " -s:v " + str(width) + 'x' + str(height) + \
              " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path
    os.system(command)
