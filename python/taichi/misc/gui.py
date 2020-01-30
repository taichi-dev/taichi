import numbers
import numpy as np
import ctypes

class GUI:
  def __init__(self, name, res=512, background_color=0x0):
    import taichi as ti
    self.name = name
    if isinstance(res, numbers.Number):
      res = (res, res)
    self.res = res
    self.core = ti.core.GUI(name, ti.veci(*res))
    self.canvas = self.core.get_canvas()
    self.background_color = background_color
    self.clear()
    
  def clear(self, color=None):
    if color is None:
      color = self.background_color
    self.canvas.clear(color)
  
  def set_image(self, img):
    import numpy as np
    import taichi as ti
    if isinstance(img, ti.Matrix):
      img = img.to_numpy(as_vector=True)
    if isinstance(img, ti.Expr):
      img = img.to_numpy()
    assert isinstance(img, np.ndarray)
    assert len(img.shape) in [2, 3]
    img = img.astype(np.float32)
    if len(img.shape) == 2:
      img = img[..., None]
    if img.shape[2] == 1:
      img = img + np.zeros(shape=(1, 1, 4))
    if img.shape[2] == 3:
      img = np.concatenate([
        img,
        np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.float32)
      ],
        axis=2)
    img = img.astype(np.float32)
    assert img.shape[:2] == self.res, "Image resolution does not match GUI resolution"
    self.core.set_img(np.ascontiguousarray(img).ctypes.data)
    
  def circle(self, pos, color, radius=1):
    import taichi as ti
    self.canvas.circle(ti.vec(pos[0],
                         pos[1])).radius(radius).color(color).finish()
    
  def circles(self, pos, color=0xFFFFFF, radius=1):
    n = pos.shape[0]
    if len(pos.shape) == 3:
      assert pos.shape[2] == 1
      pos = pos[:, :, 0]
      
    assert pos.shape == (n, 2)
    pos = np.ascontiguousarray(pos.astype(np.float32))
    pos = int(pos.ctypes.data)
    
    if isinstance(color, np.ndarray):
      assert color.shape == (n,)
      color = np.ascontiguousarray(color.astype(np.uint32))
      color_array = int(color.ctypes.data)
      color_single = 0
    elif isinstance(color, int):
      color_array = 0
      color_single = color
    else:
      raise ValueError('Color must be an ndarray or int (e.g., 0x956333)')

    if isinstance(radius, np.ndarray):
      assert radius.shape == (n,)
      radius = np.ascontiguousarray(radius.astype(np.float32))
      radius_array = int(radius.ctypes.data)
      radius_single = 0
    elif isinstance(radius, numbers.Number):
      radius_array = 0
      radius_single = radius
    else:
      raise ValueError('Radius must be an ndarray or float (e.g., 0.4)')
    
    self.canvas.circles_batched(n, pos, color_single, color_array, radius_single, radius_array)
    
  def show(self, file=None):
    self.core.update()
    if file:
      self.core.screenshot(file)
    self.clear(self.background_color)
