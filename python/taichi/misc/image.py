import numpy as np
import taichi as ti


def imwrite(img, filename):
    if not isinstance(img, np.ndarray):
        img = img.to_numpy()

    if img.dtype in [np.uint16, np.uint32, np.uint64]:
        img = (img // (np.iinfo(img.dtype).max / 256)).astype(np.uint8)
    elif img.dtype in [np.float32, np.float64]:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f'Data type {img.dtype} not supported in ti.imwrite')

    assert len(img.shape) in [2,
                              3], "Image must be either RGB/RGBA or greyscale"
    assert img.shape[2] in [1, 3,
                            4], "Image must be either RGB/RGBA or greyscale"

    resx, resy = img.shape[:2]
    if len(img.shape) == 2:
        comp = 1
    else:
        comp = img.shape[2]

    img = np.ascontiguousarray(img.swapaxes(0, 1)[::-1, :, :])
    ptr = img.ctypes.data
    ti.core.imwrite(filename, ptr, resx, resy, comp)


def imread(filename, channels=0):
    ptr, resx, resy, comp = ti.core.imread(filename, channels)
    img = np.ndarray(shape=(resy, resx, comp), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    # TODO(archibate): Figure out how np.ndarray constructor works and replace:
    ti.core.C_memcpy(img.ctypes.data, ptr, resx * resy * comp)
    # Discussion: https://github.com/taichi-dev/taichi/issues/802
    return img.swapaxes(0, 1)[:, ::-1, :]


def imshow(img, window_name='Taichi'):
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
