import sys
import os
import shutil
import time
import random
from taichi.tools.video import make_video, interpolate_frames, mp4_to_gif, scale_video, crop_video, accelerate_video


def test_python():
  print("\nRunning python tests...\n")
  import taichi as ti
  import pytest
  if ti.is_release():
    return int(pytest.main([os.path.join(ti.package_root(), 'tests')]))
  else:
    return int(pytest.main([os.path.join(ti.get_repo_directory(), 'tests')]))


def test_cpp():
  import taichi as ti
  if not ti.core.with_cuda():
    print("Skipping legacy tests (no GPU support)")
    return 0
  # Cpp tests use the legacy non LLVM backend
  os.environ['TI_LLVM'] = '0'
  ti.reset()
  print("Running C++ tests...")
  task = ti.Task('test')
  return task.run(*sys.argv[2:])


def main(debug=False):
  lines = []
  print()
  lines.append(u' *******************************************')
  lines.append(u' **     Taichi Programming Language       **')
  lines.append(u' *******************************************')
  if debug:
    lines.append(u' *****************Debug Mode****************')
    os.environ['TI_DEBUG'] = '1'
  print(u'\n'.join(lines))
  print()
  import taichi as ti

  ti.tc_core.set_core_debug(debug)

  argc = len(sys.argv)
  if argc == 1 or sys.argv[1] == 'help':
    print(
        "    Usage: ti run [task name]        |-> Run a specific task\n"
        "           ti benchmark              |-> Run performance benchmark\n"
        "           ti test                   |-> Run all tests\n"
        "           ti test_python            |-> Run python tests\n"
        "           ti test_cpp               |-> Run cpp tests\n"
        "           ti format                 |-> Reformat modified source files\n"
        "           ti format_all             |-> Reformat all source files\n"
        "           ti build                  |-> Build C++ files\n"
        "           ti video                  |-> Make a video using *.png files in the current folder\n"
        "           ti video_scale            |-> Scale video resolution \n"
        "           ti video_crop             |-> Crop video\n"
        "           ti video_speed            |-> Speed up video\n"
        "           ti gif                    |-> Convert mp4 file to gif\n"
        "           ti doc                    |-> Build documentation\n"
        "           ti release                |-> Make source code release\n"
        "           ti debug [script.py]      |-> Debug script\n")
    exit(0)
  mode = sys.argv[1]

  t = time.time()
  if mode.endswith('.py'):
    import subprocess
    subprocess.call([sys.executable] + sys.argv[1:])
  elif mode == "run":
    if argc <= 2:
      print("Please specify [task name], e.g. test_math")
      exit(-1)
    name = sys.argv[2]
    task = ti.Task(name)
    task.run(*sys.argv[3:])
  elif mode == "debug":
    ti.core.set_core_trigger_gdb_when_crash(True)
    if argc <= 2:
      print("Please specify [file name], e.g. render.py")
      exit(-1)
    name = sys.argv[2]
    with open(name) as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
  elif mode == "test_python":
    return test_python()
  elif mode == "test_cpp":
    return test_cpp()
  elif mode == "test":
    if test_python() != 0:
      return -1
    return test_cpp()
  elif mode == "build":
    ti.core.build()
  elif mode == "format":
    ti.core.format()
  elif mode == "format_all":
    ti.core.format(all=True)
  elif mode == "statement":
    exec(sys.argv[2])
  elif mode == "update":
    ti.core.update(True)
    ti.core.build()
  elif mode == "asm":
    fn = sys.argv[2]
    os.system(r"sed '/^\s*\.\(L[A-Z]\|[a-z]\)/ d' {0} > clean_{0}".format(fn))
  elif mode == "interpolate":
    interpolate_frames('.')
  elif mode == "amal":
    cwd = os.getcwd()
    os.chdir(ti.get_repo_directory())
    with open('misc/amalgamate.py') as script:
      script = script.read()
    exec(script, {'__name__': '__main__'})
    os.chdir(cwd)
    shutil.copy(
        os.path.join(ti.get_repo_directory(), 'build/taichi.h'), './taichi.h')
  elif mode == "doc":
    os.system('cd {}/docs && sphinx-build -b html . build'.format(ti.get_repo_directory()))
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
    ti.info('Making video using {} png files...', len(files))
    ti.info("frame_rate={}", frame_rate)
    output_fn = 'video.mp4'
    make_video(files, output_path=output_fn, frame_rate=frame_rate)
    ti.info('Done! Output video file = {}', output_fn)
  elif mode == "video_scale":
    input_fn = sys.argv[2]
    assert input_fn[-4:] == '.mp4'
    output_fn = input_fn[:-4] + '-scaled.mp4'
    ratiow = float(sys.argv[3])
    if len(sys.argv) >= 5:
      ratioh = float(sys.argv[4])
    else:
      ratioh = ratiow
    scale_video(input_fn, output_fn, ratiow, ratioh)
  elif mode == "video_crop":
    if len(sys.argv) != 7:
      print('Usage: ti video_crop fn x_begin x_end y_begin y_end')
      exit(-1)
    input_fn = sys.argv[2]
    assert input_fn[-4:] == '.mp4'
    output_fn = input_fn[:-4] + '-cropped.mp4'
    x_begin = float(sys.argv[3])
    x_end = float(sys.argv[4])
    y_begin = float(sys.argv[5])
    y_end = float(sys.argv[6])
    crop_video(input_fn, output_fn, x_begin, x_end, y_begin, y_end)
  elif mode == "video_speed":
    if len(sys.argv) != 4:
      print('Usage: ti video_speed fn speed_up_factor')
      exit(-1)
    input_fn = sys.argv[2]
    assert input_fn[-4:] == '.mp4'
    output_fn = input_fn[:-4] + '-sped.mp4'
    speed = float(sys.argv[3])
    accelerate_video(input_fn, output_fn, speed)
  elif mode == "gif":
    input_fn = sys.argv[2]
    assert input_fn[-4:] == '.mp4'
    output_fn = input_fn[:-4] + '.gif'
    ti.info('Converting {} to {}'.format(input_fn, output_fn))
    framerate = 24
    mp4_to_gif(input_fn, output_fn, framerate)
  elif mode == "convert":
    # http://www.commandlinefu.com/commands/view/3584/remove-color-codes-special-characters-with-sed
    # TODO: Windows support
    for fn in sys.argv[2:]:
      print("Converting logging file: {}".format(fn))
      tmp_fn = '/tmp/{}.{:05d}.backup'.format(fn, random.randint(0, 10000))
      shutil.move(fn, tmp_fn)
      command = r'sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g"'
      os.system('{} {} > {}'.format(command, tmp_fn, fn))
  elif mode == "release":
    from git import Git
    import zipfile
    import hashlib
    g = Git(ti.get_repo_directory())
    g.init()
    with zipfile.ZipFile('release.zip', 'w') as zip:
      files = g.ls_files().split('\n')
      os.chdir(ti.get_repo_directory())
      for f in files:
        if not os.path.isdir(f):
          zip.write(f)
    ver = ti.__version__
    md5 = hashlib.md5()
    with open('release.zip', "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
        md5.update(chunk)
    md5 = md5.hexdigest()
    commit = ti.core.get_commit_hash()[:8]
    fn = f'taichi-src-v{ver[0]}-{ver[1]}-{ver[2]}-{commit}-{md5}.zip'
    import shutil
    shutil.move('release.zip', fn)
  else:
    name = sys.argv[1]
    print('Running task [{}]...'.format(name))
    task = ti.Task(name)
    task.run(*sys.argv[2:])
  print()
  print(">>> Running time: {:.2f}s".format(time.time() - t))


def main_debug():
  main(debug=True)

if __name__ == '__main__':
  exit(main())
