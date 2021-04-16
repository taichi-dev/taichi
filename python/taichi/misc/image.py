import numpy as np
from taichi.core import ti_core as _ti_core

import taichi as ti


def cook_image_to_bytes(img):
    """
    Takes a NumPy array or Taichi field of any type.
    Returns a NumPy array of uint8.
    This is used by ti.imwrite and ti.imdisplay.
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


def imdisplay(img):
    """
    Try to display image in interactive shell.
    """
    try:
        get_ipython()
    except:
        ti.imshow(img)
    else:
        from io import BytesIO

        import IPython.display
        import PIL.Image
        img = cook_image_to_bytes(img)
        with BytesIO() as f:
            PIL.Image.fromarray(img).save(f, 'png')
            IPython.display.display(IPython.display.Image(data=f.getvalue()))


def imresize(img, w, h=None):
    """
    Resize an image to a specific size.
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
    """
    Save image to a specific file.
    """
    img = cook_image_to_bytes(img)
    img = np.ascontiguousarray(img)
    ptr = img.ctypes.data
    resy, resx, comp = img.shape
    _ti_core.imwrite(filename, ptr, resx, resy, comp)


def imread(filename, channels=0):
    """
    Load image from a specific file.
    """
    ptr, resx, resy, comp = _ti_core.imread(filename, channels)
    img = np.ndarray(shape=(resy, resx, comp), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    # TODO(archibate): Figure out how np.ndarray constructor works and replace:
    _ti_core.C_memcpy(img.ctypes.data, ptr, resx * resy * comp)
    # Discussion: https://github.com/taichi-dev/taichi/issues/802
    return img.swapaxes(0, 1)[:, ::-1, :]


def imshow(img, window_name='imshow'):
    """
    Show image in a Taichi GUI.
    """
    if not isinstance(img, np.ndarray):
        img = img.to_numpy()
    assert len(img.shape) in [2,
                              3], "Image must be either RGB/RGBA or greyscale"

    with ti.GUI(window_name, res=img.shape[:2]) as gui:
        img = gui.cook_image(img)
        while gui.running:
            if gui.get_event(ti.GUI.ESCAPE):
                gui.running = False

            gui.set_image(img)
            gui.show()


__all__ = [
    'imshow',
    'imread',
    'imwrite',
    'imresize',
    'imdisplay',
]
