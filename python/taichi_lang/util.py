def is_taichi_class(rhs):
  taichi_class = False
  try:
    if rhs.is_taichi_class:
      taichi_class = True
  except:
    pass
  return taichi_class

