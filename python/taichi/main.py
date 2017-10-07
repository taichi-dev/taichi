import taichi
import sys


class Task:

  def run(self, *args):
    pass


def create_task():
  return Task()


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
    task_name = sys.argv[2]
    task = create_task(task_name)
    task.run(sys.argv[2:])
  elif mode == "test":
    print("not implemented")
    exit(-1)
  else:
    print("Mode should be 'run' or 'test' instead of " + mode)
    exit(-1)
