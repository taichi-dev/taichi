from tk.viewer import ImageViewer, update_tk
import taichi as tc
import numpy as np

viewers = {}


def show_image(name, img):
  # Ensures img is w*h*3 (RGB)
  if isinstance(img, tc.core.Array2Dreal):
    img = tc.util.array2d_to_ndarray(img)
  if isinstance(img, tc.core.Array2DVector3):
    img = tc.util.array2d_to_ndarray(img)
  if isinstance(img, tc.core.Array2DVector4):
    img = tc.util.array2d_to_ndarray(img)[:, :, :3]
  if isinstance(img, np.ndarray):
    if len(img.shape) == 2:
      img = img[:, :, None] * np.ones((1, 1, 3), dtype='uint8')
    else:
      if img.shape[2] == 1:
        img = np.outer(img, np.ones((1, 1, 3), dtype='float32'))
      else:
        img = img[:, :, :3]
  img = (img * 255).astype('uint8')

  if name in viewers:
    viewers[name].update(img)
  else:
    viewers[name] = ImageViewer(name, img)
  update_tk()


# TODO: destory viewers atexit
