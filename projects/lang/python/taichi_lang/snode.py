from .core import taichi_lang_core

class SNode:
  def __init__(self, ptr):
    self.ptr = ptr

  def dense(self, indices, dimensions):
    if isinstance(dimensions, int):
      dimensions = [dimensions] * len(indices)
    return SNode(self.ptr.dense(indices, dimensions))

  def dynamic(self, index, dimension):
    assert len(index) == 1
    return SNode(self.ptr.dynamic(index[0], dimension))

  def pointer(self):
    return SNode(self.ptr.pointer())

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
