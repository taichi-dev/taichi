from taichi import Task
import sys


def main():
  lines = []
  print()
  lines.append(u'{:^43}'.format(u' '.join([u'\u262f'] * 8)))
  lines.append(u' ******************************************')
  lines.append(u' ** Taichi - A Computer Graphics Library **')
  lines.append(u' ******************************************')
  lines.append(u'{:^43}'.format(u"\u2630 \u2631 \u2632 \u2633 "
                    "\u2634 \u2635 \u2636 \u2637"))
  print(u'\n'.join(lines))
  print()

  argc = len(sys.argv)
  if argc == 1:
    print("    Usage: ti run [task name] \n"
          "           ti test\n"
          "           ti build\n"
          "           ti update\n"
          "           ti *.py [arguments]\n")
    exit(-1)
  mode = sys.argv[1]

  if mode.endswith('.py'):
    with open(mode) as script:
      exec(script.read())
    exit()

  if mode == "run":
    if argc <= 2:
      print("Please specify [task name], e.g. test_math")
      exit(-1)
    name = sys.argv[2]
    task = Task(name)
    task.run(sys.argv[3:])
  elif mode == "test":
    task = Task('test')
    task.run(sys.argv[2:])
    exit(-1)
  else:
    print("Unknown command '{}'".format(mode))
    exit(-1)

if __name__ == '__main__':
  main()
