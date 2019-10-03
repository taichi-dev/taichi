import sys
import os
import shutil
import time
import random
from taichi.tools.video import make_video, interpolate_frames
from taichi.core.util import get_projects, activate_package, deactivate_package

packages = {
  'mpm': 'https://github.com/yuanming-hu/taichi_mpm',
  'wushi': 'https://github.com/yuanming-hu/taichi_wushi'
}

def run_pytest():
  print("\nRunning python tests...\n")
  from pathlib import Path
  import taichi as ti
  import subprocess
  hooks = list(filter(lambda x: str(x).endswith('testhook.py'), Path(ti.get_project_directory()).rglob("*.py")))
  for hook in map(str, hooks):
    print("\nRunning testhook {}...\n".format(hook))
    cwd = os.getcwd()
    os.chdir(hook[:hook.rfind('/')])
    subprocess.call([sys.executable, hook])
    os.chdir(cwd)

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

def main(debug=False):
  lines = []
  print()
  lines.append(u'{:^43}'.format(u' '.join([u'\u262f'] * 8)))
  lines.append(u' *******************************************')
  lines.append(u' **                Taichi                 **')
  lines.append(u' **                ~~~~~~                 **')
  lines.append(u' ** High-Performance Programming Language **')
  lines.append(u' *******************************************')
  lines.append(u'{:^43}'.format(u"\u2630 \u2631 \u2632 \u2633 "
                                "\u2634 \u2635 \u2636 \u2637"))
  print(u'\n'.join(lines))
  print()
  import taichi as tc

  tc.tc_core.set_core_debug(debug)

  argc = len(sys.argv)
  if argc == 1 or sys.argv[1] == 'help':
    print(
      "    Usage: ti run [task name]        |-> Run a specific task\n"
      "           ti test                   |-> Run tests\n"
      "           ti install                |-> Install package\n"
      "           ti proj                   |-> List all projects\n"
      "           ti proj activate [name]   |-> Activate project\n"
      "           ti proj deactivate [name] |-> Deactivate project\n"
      "           ti build                  |-> Build C++ files\n"
      "           ti amal                   |-> Generate amalgamated taichi.h\n"
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
      "           ti doc                    |-> Build documentation\n"
      "           ti merge                  |-> Merge images in folders horizontally\n"
      "           ti debug [script.py]      |-> Debug script\n")
    exit(-1)
  mode = sys.argv[1]

  t = time.time()
  if mode.endswith('.py'):
    with open(mode) as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
  elif mode.endswith('.cpp'):
    command = 'g++ {} -o {} -g -std=c++14 -O3 -lX11 -lpthread'.format(mode, mode[:-4])
    print(command)
    ret = os.system(command)
    if ret == 0:
      os.system('./{}'.format(mode[:-4]))
  elif mode == "run":
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
    if len(sys.argv) == 2:
      run_pytest()
    print("Running C++ tests...")
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
  elif mode == "install":
    os.chdir(tc.get_directory('projects'))
    pkg = sys.argv[2]
    if pkg not in packages:
      url = pkg
      pkg = pkg.split('/')[-1]
    else:
      url = packages[pkg]
    tc.info('Cloning and building package {}...'.format(pkg))
    os.system('git clone {} {}'.format(url, pkg))
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
  elif mode == "amal":
    cwd = os.getcwd()
    os.chdir(tc.get_repo_directory())
    with open('misc/amalgamate.py') as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
    os.chdir(cwd)
    shutil.copy(os.path.join(tc.get_repo_directory(), 'build/taichi.h'), './taichi.h')
  elif mode == "doc":
    os.system('cd docs && sphinx-build -b html . build')
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
  elif mode == "merge":
    import cv2 # TODO: remove this dependency
    import numpy as np
    folders = sys.argv[2:]
    os.makedirs('merged', exist_ok=True)
    for fn in sorted(os.listdir(folders[0])):
      imgs = []
      for fld in folders:
        img = cv2.imread(os.path.join(fld, fn))
        imgs.append(img)
      img = np.hstack(imgs)
      cv2.imwrite(os.path.join('merged', fn), img)
  else:
    name = sys.argv[1]
    print('Running task [{}]...'.format(name))
    task = tc.Task(name)
    task.run(*sys.argv[2:])
  print()
  print(">>> Running time: {:.2f}s".format(time.time() - t))

if __name__ == '__main__':
  main()
