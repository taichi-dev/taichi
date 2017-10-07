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
  
  def run(self, *args):
    self.c.run(*args)


def main():
  print("                  \u262f \u262f \u262f \u262f\n"
        " ******************************************\n"
        " ** Taichi - A Computer Graphics Library **\n"
        " ******************************************\n"
        "              \u2630 \u2631 \u2632 \u2633 "
        "\u2634 \u2635 \u2636 \u2637\n")

  argc = len(sys.argv)
  if argc == 1:
    print("    Usage: ti run  [task name] \n"
          "           ti test [module name]")
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
