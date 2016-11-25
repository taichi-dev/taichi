import os

def make_video(input_files, width, height, output_path):
    command = "ffmpeg -framerate 24 -i " + input_files + \
              " -s:v " + str(width) + 'x' + str(height) + \
              " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path
    os.system(command)
