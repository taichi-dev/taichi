from taichi.core import unit


@unit('task')
class Task:

  def run(self, *args):
    return self.c.run(args)
