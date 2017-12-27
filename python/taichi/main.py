import taichi as tc
import sys
import os
import shutil
import random


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
  if argc == 1 or sys.argv[1] == 'help':
    print(
        "    Usage: ti run [task name]        |-> Run a specific task\n"
        "           ti test                   |-> Run tests\n"
        "           ti build                  |-> Build C++ files\n"
        "           ti update                 |-> Update taichi and projects\n"
        "           ti format                 |-> Format taichi and projects\n"
        "                                         (C++ source and python scripts)\n"
        "           ti *.py [arguments]       |-> Run scripts\n")
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
    task = tc.Task(name)
    task.run(sys.argv[3:])
  elif mode == "test":
    # tc.core.set_core_trigger_gdb_when_crash(True)
    task = tc.Task('test')
    task.run(sys.argv[2:])
  elif mode == "build":
    tc.core.build()
  elif mode == "format":
    tc.core.format()
  elif mode == "update":
    tc.core.update(True)
    tc.core.build()
  elif mode == "convert":
    # http://www.commandlinefu.com/commands/view/3584/remove-color-codes-special-characters-with-sed
    # TODO: Windows support
    for fn in sys.argv[2:]:
      print("Converting logging file: {}".format(fn))
      tmp_fn = '/tmp/{}.{:05d}.backup'.format(fn, random.randint(0, 10000))
      shutil.move(fn, tmp_fn)
      command = r'sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"'
      os.system('{} {} > {}'.format(command, tmp_fn, fn))
  else:
    print("Unknown command '{}'".format(mode))
    exit(-1)


if __name__ == '__main__':
  main()
