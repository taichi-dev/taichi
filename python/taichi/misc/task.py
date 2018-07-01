from taichi.core import unit

@unit('task')
class Task:
  def run(self, *args):
    self.c.run(args)

