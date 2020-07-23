import numpy as np
import taichi as ti


def cook_image_to_bytes(img):
    """
    Takes a NumPy array or Taichi tensor of any type.
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
    if ti.lang.shell.oinspect.name == ti.lang.shell.ShellType.JUPYTER:
        import PIL.Image
        from io import BytesIO
        import IPython.display
        import numpy as np
        img = cook_image_to_bytes(img)
        with BytesIO() as f:
            PIL.Image.fromarray(img).save(f, 'png')
            IPython.display.display(IPython.display.Image(data=f.getvalue()))
    else:
        ti.imshow(img)


def imwrite(img, filename):
    """
    Save image to a specific file.
    """
    img = cook_image_to_bytes(img)
    img = np.ascontiguousarray(img)
    ptr = img.ctypes.data
    resy, resx, comp = img.shape
    ti.core.imwrite(filename, ptr, resx, resy, comp)


def imread(filename, channels=0):
    """
    Load image from a specific file.
    """
    ptr, resx, resy, comp = ti.core.imread(filename, channels)
    img = np.ndarray(shape=(resy, resx, comp), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    # TODO(archibate): Figure out how np.ndarray constructor works and replace:
    ti.core.C_memcpy(img.ctypes.data, ptr, resx * resy * comp)
    # Discussion: https://github.com/taichi-dev/taichi/issues/802
    return img.swapaxes(0, 1)[:, ::-1, :]


def imshow(img, window_name='Taichi'):
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
    'imdisplay',
]
