from .core import taichi_lang_core

class SNode:
  def __init__(self, ptr):
    self.ptr = ptr

  def dense(self, *args):
    return SNode(self.ptr.dense(*args))

  def pointer(self):
    return SNode(self.ptr.pointer())

  def place(self, *args):
    self.ptr.place(*args)
    return self
