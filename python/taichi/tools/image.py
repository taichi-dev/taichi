import numpy as np
from taichi._lib import core as _ti_core

import taichi as ti


def cook_image_to_bytes(img):
    """
    Takes a NumPy array or Taichi field of any type.
    Returns a NumPy array of uint8.
    This is used by ti.imwrite.
    """
    if not isinstance(img, np.ndarray):
        img = img.to_numpy()

    if img.dtype in [np.uint16, np.uint32, np.uint64]:
        img = (img // (np.iinfo(img.dtype).max // 256)).astype(np.uint8)
    elif img.dtype in [np.float32, np.float64]:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f'Data type {img.dtype} not supported in ti.imwrite')

    assert len(img.shape) in [2,
                              3], "Image must be either RGB/RGBA or greyscale"

    if len(img.shape) == 2:
        img = img.reshape(*img.shape, 1)

    assert img.shape[2] in [1, 3,
                            4], "Image must be either RGB/RGBA or greyscale"

    return img.swapaxes(0, 1)[::-1, :]


def imresize(img, w, h=None):
    """Resize an image to a specified size.

    Args:
        img (Union[:class:`~taichi.lang.field.Field`, `np.ndarray`]): A field or an numpy ndarray with shape `(width, height, ...)`.
        w (int): The output image width after resize.
        h (int, optional): The output image height after resize, will be the same as `w` if not set. Default to `None`.

    Returns:
        np.ndarray: An output image after resize input.
    """
    if not isinstance(img, np.ndarray):
        img = img.to_numpy()
    if h is None:
        h = w
    if (w, h) == img.shape[:2]:
        return img
    assert isinstance(w, int) and isinstance(h, int) and w > 1 and h > 1
    u, v = (img.shape[0]) / (w), (img.shape[1]) / (h)
    x = np.clip(np.arange(w) * u, 0, img.shape[0] - 1).astype(np.int32)
    y = np.clip(np.arange(h) * v, 0, img.shape[1] - 1).astype(np.int32)
    return img[tuple(np.meshgrid(x, y))].swapaxes(0, 1)


def imwrite(img, filename):
    """Save a field or an array as an image.

    Args:
        img (Union[:class:`taichi.lang.field.Field`, np.ndarray]): A field or an numpy.ndarray, with shape \
            `(height, width)` or `(height, width, 3)` or `(height, width, 4)`, \
            if dtype is float-type (`ti.f16`, `ti.f32`, `np.float32` etc), **the elements in the array should be float in range \[0.0, 1.0\]**. \
                Otherwise `ti.imwrite` will clamp them into \[0.0, 1.0\]\
                if dtype is int-type (`ti.u8`, `ti.u16`, `np.uint8` etc), , **the value of each pixel can be any valid integer in its own bounds**. These integers in this field will be scaled to \[0, 255\] by being divided over the upper bound of its basic type accordingly.
        filename (str): The filename to save to.
    """
    img = cook_image_to_bytes(img)
    img = np.ascontiguousarray(img)
    ptr = img.ctypes.data
    resy, resx, comp = img.shape
    _ti_core.imwrite(filename, ptr, resx, resy, comp)


def imread(filename, channels=0):
    """Read an image from a file as an numpy.ndarray.

    Args:
        filename (str): The file name to be read.
        channels (int, optinal): Number of channels in the image, default to 0.

    Returns:
        np.ndarray: The `numpy.ndarray` obtained by reading the image.
    """
    ptr, resx, resy, comp = _ti_core.imread(filename, channels)
    img = np.ndarray(shape=(resy, resx, comp), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    # TODO(archibate): Figure out how np.ndarray constructor works and replace:
    _ti_core.C_memcpy(img.ctypes.data, ptr, resx * resy * comp)
    # Discussion: https://github.com/taichi-dev/taichi/issues/802
    return img.swapaxes(0, 1)[:, ::-1, :]


def imshow(img, title='imshow'):
    """Display a `taichi.field` or an `numpy.ndarray` in a Taichi GUI window or an interactive Ipython notebook.

    Args:
        img (Union[:class:`taichi.lang.field.Field`, np.ndarray]): A field or an numpy.ndarray with shape \
            `(width, height)` or `(height, width, 3)` or `(height, width, 4)`.

        title (str, optional): The title of GUI window. Default to `imshow`.
    """
    try:  # check if we are in Ipython environment
        get_ipython()
    except:
        if not isinstance(img, np.ndarray):
            img = img.to_numpy()
            assert len(
                img.shape) in [2,
                               3], "Image must be either RGB/RGBA or greyscale"

        with ti.GUI(title, res=img.shape[:2]) as gui:
            img = gui.cook_image(img)
            while gui.running:
                if gui.get_event(ti.GUI.ESCAPE):
                    gui.running = False

                gui.set_image(img)
                gui.show()
    else:
        import IPython.display  # pylint: disable=C0415
        import PIL.Image  # pylint: disable=C0415
        img = cook_image_to_bytes(img)
        IPython.display.display(PIL.Image.fromarray(img))


__all__ = ['imread', 'imresize', 'imshow', 'imwrite']
