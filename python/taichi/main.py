import taichi
from taichi import core
from taichi.core.unit import unit
from taichi.misc.util import config_from_dict
import sys


@unit('task')
class Task:
  def __init__(self, name, **kwargs):
    self.c = core.create_task(name)
    self.c.initialize(config_from_dict(kwargs))
    print("initializing task", name)
  
  def run(self, *args):
    self.c.run(*args)


def main():
  print("                          *******                          \n"
        " ********************************************************* \n"
        " ** Taichi - Physically based Computer Graphics Library ** \n"
        " ********************************************************* \n"
        "                          *******                          \n")

  argc = len(sys.argv)
  if argc == 1:
    print("    usage: taichi run [task name] \n"
          "           taichi test [module name]")
    exit(-1)
  mode = sys.argv[1]
  if mode == "run":
    if argc <= 2:
      print("Please specify [task name], e.g. test_math")
      exit(-1)
    name = sys.argv[2]
    task = Task(name)
    task.run(sys.argv[3:])
  elif mode == "test":
    print("not implemented")
    exit(-1)
  else:
    print("Mode should be 'run' or 'test' instead of " + mode)
    exit(-1)
