import numbers

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
    
  def circles(self, pos, color, radius=1):
    import taichi as ti
    for i in range(len(pos)):
      self.canvas.circle(ti.vec(pos[i, 0],
                                pos[i, 1])).radius(radius).color(color[i]).finish()
    
  def show(self, file=None):
    self.core.update()
    if file:
      self.core.screenshot(file)
    self.clear(self.background_color)
