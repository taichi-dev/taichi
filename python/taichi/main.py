import sys
import os
import shutil
import random
from taichi.tools.video import make_video, interpolate_frames
from taichi.core.util import get_projects, activate_package, deactivate_package

def print_all_projects():
  from colorama import Fore, Back, Style
  print(Fore.GREEN, end='')
  print("Active projects:")
  projs = get_projects(active=True)
  for p in projs:
    print('    {}'.format(p))

  print(Fore.BLUE, end='')
  projs = get_projects(active=False)
  print("Inactive projects:")
  for p in projs:
    print('    {}'.format(p))
  print(Fore.RESET, end='')

def plot(fn):
  import matplotlib.pyplot as plt
  with open(fn) as f:
    lines = f.readlines()
    T = []
    M = []
    for l in lines:
      t, m = map(float, l.split(' '))
      T.append(t)
      M.append(m / 2 ** 30)

  base_m = min(M)
  base_t = T[0]
  for i in range(len(T)):
    T[i] -= base_t
    M[i] -= base_m

  plt.clf()
  plt.plot(T, M)
  plt.xlabel('Time (seconds)')
  plt.ylabel('Memory Consumption (G Bytes)')
  title = "Max={:3f} GB (taichi runtime={:3f} GB)".format(max(M), base_m)
  plt.title(title)
  plt.ylim(0, max(M) * 1.2)
  plt.show()

def main():
  lines = []
  print()
  lines.append(u'{:^43}'.format(u' '.join([u'\u262f'] * 8)))
  lines.append(u' *******************************************')
  lines.append(u' **                Taichi                 **')
  lines.append(u' **                ~~~~~~                 **')
  lines.append(u' ** Open Source Computer Graphics Library **')
  lines.append(u' *******************************************')
  lines.append(u'{:^43}'.format(u"\u2630 \u2631 \u2632 \u2633 "
                                "\u2634 \u2635 \u2636 \u2637"))
  print(u'\n'.join(lines))
  print()
  import taichi as tc

  argc = len(sys.argv)
  if argc == 1 or sys.argv[1] == 'help':
    print(
        "    Usage: ti run [task name]        |-> Run a specific task\n"
        "           ti test                   |-> Run tests\n"
        "           ti daemon                 |-> Start daemon process\n"
        "           ti proj                   |-> List all projects\n"
        "           ti proj activate [name]   |-> Activate project\n"
        "           ti proj deactivate [name] |-> Deactivate project\n"
        "           ti build                  |-> Build C++ files\n"
        "           ti clean asm [*.s]        |-> Clean up gcc ASM\n"
        "           ti plot [*.txt]           |-> Plot a memory usage curve\n"
        "           ti update                 |-> Update taichi and projects\n"
        "           ti video                  |-> Make a video using *.png files in the current folder\n"
        "           ti convert                |-> Delete color controllers in a log file\n"
        "           ti exec                   |-> Invoke a executable in the 'build' folder\n"
        "           ti format                 |-> Format taichi and projects\n"
        "                                         (C++ source and python scripts)\n"
        "           ti statement [statement]  |-> Execute a single statement (with taichi imported as tc\n"
        "           ti [script.py]            |-> Run script\n"
        "           ti debug [script.py]      |-> Debug script\n")
    exit(-1)
  mode = sys.argv[1]

  if mode.endswith('.py'):
    with open(mode) as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
    exit()

  if mode == "run":
    if argc <= 2:
      print("Please specify [task name], e.g. test_math")
      exit(-1)
    name = sys.argv[2]
    task = tc.Task(name)
    task.run(*sys.argv[3:])
  elif mode == "debug":
    tc.core.set_core_trigger_gdb_when_crash(True)
    if argc <= 2:
      print("Please specify [file name], e.g. render.py")
      exit(-1)
    name = sys.argv[2]
    with open(name) as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
    exit()
  elif mode == "daemon":
    from taichi.system.daemon import start
    if len(sys.argv) > 2:
      # Master daemon
      start(True)
    else:
      # Slave daemon
      start(False)
  elif mode == "proj":
    if len(sys.argv) == 2:
      print_all_projects()
    elif sys.argv[2] == 'activate':
      proj = sys.argv[3]
      activate_package(proj)
    elif sys.argv[2] == 'deactivate':
      proj = sys.argv[3]
      deactivate_package(proj)
    else:
      assert False
  elif mode == "test":
    task = tc.Task('test')
    task.run(*sys.argv[2:])
  elif mode == "build":
    tc.core.build()
  elif mode == "format":
    tc.core.format()
  elif mode == "statement":
    exec(sys.argv[2])
  elif mode == "plot":
    plot(sys.argv[2])
  elif mode == "update":
    tc.core.update(True)
    tc.core.build()
  elif mode == "asm":
    fn = sys.argv[2]
    os.system(r"sed '/^\s*\.\(L[A-Z]\|[a-z]\)/ d' {0} > clean_{0}".format(fn))
  elif mode == "exec":
    import subprocess
    exec_name = sys.argv[2]
    folder = tc.get_bin_directory()
    assert exec_name in os.listdir(folder)
    subprocess.call([os.path.join(folder, exec_name)] + sys.argv[3:])
  elif mode == "interpolate":
    interpolate_frames('.')
  elif mode == "video":
    files = sorted(os.listdir('.'))
    files = list(filter(lambda x: x.endswith('.png'), files))
    if len(sys.argv) >= 3:
      frame_rate = int(sys.argv[2])
    else:
      frame_rate = 24
    if len(sys.argv) >= 4:
      trunc = int(sys.argv[3])
      files = files[:trunc]
    tc.info('Making video using {} png files...', len(files))
    tc.info("frame_rate={}", frame_rate)
    output_fn = 'video.mp4'
    make_video(files, output_path=output_fn, frame_rate=frame_rate)
    tc.info('Done! Output video file = {}', output_fn)
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
