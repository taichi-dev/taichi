class ndrange:
  def __init__(self, *args):
    import taichi as ti
    args = list(args)
    for i in range(len(args)):
      if isinstance(args[i], list):
        args[i] = list(args[i])
      if not isinstance(args[i], tuple):
        args[i] = (0, args[i])
      assert len(args[i]) == 2
      args[i] = ti.Expr(args[i][0]), ti.Expr(args[i][1])
    self.bounds = args
    
    self.dimensions = [None] * len(args)
    for i in range(len(self.bounds)):
      self.dimensions[i] = self.bounds[i][1] - self.bounds[i][0]

    self.acc_dimensions = self.dimensions.copy()
    for i in reversed(range(len(self.bounds) - 1)):
       self.acc_dimensions[i] = self.acc_dimensions[i] * self.acc_dimensions[i + 1]
