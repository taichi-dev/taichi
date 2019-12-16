class SNode:

  def __init__(self, ptr):
    self.ptr = ptr

  def dense(self, indices, dimensions):
    if isinstance(dimensions, int):
      dimensions = [dimensions] * len(indices)
    return SNode(self.ptr.dense(indices, dimensions))

  def dynamic(self, index, dimension, chunk_size=None):
    assert len(index) == 1
    if chunk_size is None:
      chunk_size = dimension
    return SNode(self.ptr.dynamic(index[0], dimension, chunk_size))

  def pointer(self):
    return SNode(self.ptr.pointer())

  def bitmasked(self, val=True):
    self.ptr.bitmasked(val)
    return self

  def place(self, *args):
    from .expr import Expr
    for arg in args:
      if isinstance(arg, Expr):
        self.ptr.place(Expr(arg).ptr)
      elif isinstance(arg, list):
        for x in arg:
          self.place(x)
      else:
        arg.place(self)
    return self

  def lazy_grad(self):
    self.ptr.lazy_grad()

  def parent(self):
    return SNode(self.ptr.snode().parent)

  def data_type(self):
    return self.ptr.data_type()

  def dim(self):
    return self.ptr.num_active_indices()

  def get_shape(self, i):
    return self.ptr.get_num_elements_along_axis(i)
