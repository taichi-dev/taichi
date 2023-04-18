import os
import shutil

from taichi._lib.utils import get_os_name
from taichi.tools.image import imwrite

FRAME_FN_TEMPLATE = "%06d.png"
FRAME_DIR = "frames"

# Write the frames to the disk and then make videos (mp4 or gif) if necessary


def scale_video(input_fn, output_fn, ratiow, ratioh):
    os.system(f'ffmpeg -i {input_fn}  -vf "scale=iw*{ratiow:.4f}:ih*{ratioh:.4f}" {output_fn}')


def crop_video(input_fn, output_fn, x_begin, x_end, y_begin, y_end):
    os.system(
        f'ffmpeg -i {input_fn} -filter:v "crop=iw*{x_end - x_begin:.4f}:ih*{y_end - y_begin:.4f}:iw*{x_begin:0.4f}:ih*{1 - y_end:0.4f}" {output_fn}'
    )


def accelerate_video(input_fn, output_fn, speed):
    os.system(f'ffmpeg -i {input_fn} -filter:v "setpts={1 / speed:.4f}*PTS" {output_fn}')


def get_ffmpeg_path():
    return "ffmpeg"


def mp4_to_gif(input_fn, output_fn, framerate):
    # Generate the palette
    palette_name = "palette.png"
    if get_os_name() == "win":
        command = get_ffmpeg_path() + f" -loglevel panic -i {input_fn} -vf 'palettegen' -y {palette_name}"
    else:
        command = (
            get_ffmpeg_path() + f" -loglevel panic -i {input_fn} -vf 'fps={framerate},"
            f"scale=320:640:flags=lanczos,palettegen' -y {palette_name}"
        )
    # print command
    os.system(command)

    # Generate the GIF
    command = get_ffmpeg_path() + f" -loglevel panic -i {input_fn} -i {palette_name} -lavfi paletteuse -y {output_fn}"
    # print command
    os.system(command)
    os.remove(palette_name)


class VideoManager:
    """Utility class for exporting results to `mp4` and `gif` formats.
    This class relies on `ffmpeg`.

    Args:
        output_dir (str): directory to save the frames.
        video_filename (str): filename for the video. default filename is video.mp4.
        width, height (int): resolution of the video.
        post_processor (object): any object that implements the `process(img)`
            method, which accepts an image as a `numpy.ndarray` and returns
            the process image.
        framerate (int): frame rate of the video.
        automatic_build (bool): automatically generate the resulting video or not.

    Example::

        >>> video_manager = ti.tools.VideoManager(output_dir="./output", framerate=24, automatic_build=False)
        >>> for i in range(50):
        >>>     render()
        >>>     img = pixels.to_numpy()
        >>>     video_manager.write_frame(img)
        >>>
        >>> video_manager.make_video(gif=True, mp4=True)

    Returns:
        An instance of :class:`taichi.tools.VideoManager` class.
    """

    def __init__(
        self,
        output_dir,
        video_filename=None,
        width=None,
        height=None,
        post_processor=None,
        framerate=24,
        automatic_build=True,
    ):
        assert (width is None) == (height is None)
        self.width = width
        self.height = height
        self.directory = output_dir
        self.frame_directory = os.path.join(self.directory, FRAME_DIR)
        try:
            os.makedirs(self.frame_directory)
        except:
            pass
        self.video_filename = video_filename
        self.next_video_checkpoint = 4
        self.framerate = framerate
        self.post_processor = post_processor
        self.frame_counter = 0
        self.frame_fns = []
        self.automatic_build = automatic_build

    def get_output_filename(self, suffix):
        if not self.video_filename:
            return os.path.join(self.directory, "video" + suffix)
        filename, extension = os.path.splitext(self.video_filename)
        if extension is not None:
            print(f"Warning: file extension {extension} will be disregarded!")
        return os.path.join(self.directory, filename + suffix)

    def write_frame(self, img):
        """Write an `numpy.ndarray` `img` to an image file.

        The filename will be automatically determined by this manager
        and the frame counter.
        """
        if img.shape[0] % 2 != 0:
            print("Warning: height is not divisible by 2! Dropping last row")
            img = img[:-1]
        if img.shape[1] % 2 != 0:
            print("Warning: width is not divisible by 2! Dropping last column")
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
        """Returns path to the directory where the image files are located in."""
        return self.frame_directory

    def write_frames(self, images):
        """Write a list of `numpy.ndarray` `images` to image files."""
        for img in images:
            self.write_frame(img)

    def clean_frames(self):
        """Delete all previous image files in the saved directory."""
        for fn in os.listdir(self.frame_directory):
            if fn.endswith(".png") and fn in self.frame_fns:
                os.remove(fn)

    def make_video(self, mp4=True, gif=True):
        """Convert the image files to a `mp4` or `gif` animation."""
        fn = self.get_output_filename(".mp4")
        command = (
            (get_ffmpeg_path() + f" -loglevel panic -framerate {self.framerate} -i ")
            + os.path.join(self.frame_directory, FRAME_FN_TEMPLATE)
            + " -s:v "
            + str(self.width)
            + "x"
            + str(self.height)
            + " -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p -y "
            + fn
        )

        os.system(command)

        if gif:
            mp4_to_gif(
                self.get_output_filename(".mp4"),
                self.get_output_filename(".gif"),
                self.framerate,
            )

        if not mp4:
            os.remove(fn)


def interpolate_frames(frame_dir, mul=4):
    # TODO: remove dependency on cv2 here
    import cv2  # pylint: disable=C0415

    files = os.listdir(frame_dir)
    images = []
    images_interpolated = []
    for f in sorted(files):
        if f.endswith("png"):
            images.append(cv2.imread(f) / 255.0)  # pylint: disable=E1101

    for i in range(len(images) - 1):
        images_interpolated.append(images[i])
        for j in range(mul - 1):
            alpha = 1 - j / mul
            images_interpolated.append(images[i] * alpha + images[i + 1] * (1 - alpha))

    images_interpolated.append(images[-1])

    os.makedirs("interpolated", exist_ok=True)
    for i, img in enumerate(images_interpolated):
        cv2.imwrite(f"interpolated/{i:05d}.png", img * 255.0)  # pylint: disable=E1101


def ffmpeg_common_args(frame_rate, input_fn, width, height, crf, output_path):
    return (
        f"{get_ffmpeg_path()} -y -loglevel panic -framerate {frame_rate} -i {input_fn} -s:v {width}x{height} "
        + f"-c:v libx264 -profile:v high -crf {crf} -pix_fmt yuv420p {output_path}"
    )


def make_video(input_files, width=0, height=0, frame_rate=24, crf=20, output_path="video.mp4"):
    """Convert a list of image files to a `gif` or `mp4` animation.

    Args:
        input_files (list[str]): the list of image file names.
        width (int): output video width.
        height (int): output video height.
        frame_rate (int): framerate of the output video.
        crf (int): quality of the output video, the lower the better quality,
            but also larger file size.
        output_path (str): path to the output video.
    """
    if isinstance(input_files, list):
        from PIL import Image  # pylint: disable=C0415

        with Image.open(input_files[0]) as img:
            width, height = img.size
        tmp_dir = "tmp_ffmpeg_dir"
        os.mkdir(tmp_dir)
        if width % 2 != 0:
            print(f"Width ({width}) not divisible by 2")
            width -= 1
        if height % 2 != 0:
            print(f"Height ({width}) not divisible by 2")
            height -= 1
        for i, inp in enumerate(input_files):
            shutil.copy(inp, os.path.join(tmp_dir, f"{i:06d}.png"))
        inputs = f"{tmp_dir}/%06d.png"
        command = ffmpeg_common_args(frame_rate, inputs, width, height, crf, output_path)
        ret = os.system(command)
        assert ret == 0, "ffmpeg failed to generate video file."
        for i in range(len(input_files)):
            os.remove(os.path.join(tmp_dir, f"{i:06d}.png"))
        os.rmdir(tmp_dir)
    elif isinstance(input_files, str):
        assert width != 0 and height != 0
        command = ffmpeg_common_args(frame_rate, input_files, width, height, crf, output_path)
        ret = os.system(command)
        assert ret == 0, "ffmpeg failed to generate video file."
    else:
        assert (
            False
        ), f'input_files should be list (of files) or str (of file template, e.g., "%04d.png") instead of {type(input_files)}'


__all__ = ["VideoManager"]
