from taichi import Task
import sys


def main():
  lines = []
  print()
  lines.append('{:^43}'.format(' '.join(['\u262f'] * 8)))
  lines.append(' ******************************************')
  lines.append(' ** Taichi - A Computer Graphics Library **')
  lines.append(' ******************************************')
  lines.append('{:^43}'.format("\u2630 \u2631 \u2632 \u2633 "
                    "\u2634 \u2635 \u2636 \u2637"))
  print(u'\n'.join(lines))
  print()

  argc = len(sys.argv)
  if argc == 1:
    print("    Usage: ti run  [task name] \n"
          "           ti test [module name]\n"
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
    print("not implemented")
    exit(-1)
  else:
    print("Mode should be 'run' or 'test' instead of '%s'" % mode)
    exit(-1)

if __name__ == '__main__':
  main()
