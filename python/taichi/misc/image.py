import numpy as np
import taichi as ti

def cook_image(img):
  if isinstance(img, ti.Matrix):
    img = img.to_numpy(as_vector=True)
  if isinstance(img, ti.Expr):
    img = img.to_numpy()
  assert isinstance(img, np.ndarray)
  assert len(img.shape) in [2, 3]
  img = img.astype(np.uint8)
  return img

def imwrite(img, filename):
  img = cook_image(img)
  resx, resy = img.shape[:2]
  if len(img.shape) == 3:
    comp = img.shape[2]
  else:
    comp = 1
  img = np.ascontiguousarray(img).ctypes.data
  ti.core.imwrite(filename, img, resx, resy, comp)

def imread(filename):
  raise NotImplementedError
