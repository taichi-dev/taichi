import sys
import os
import shutil
import time
import math
import random
import argparse
from collections import defaultdict
from colorama import Fore, Back, Style
from taichi.tools.video import make_video, interpolate_frames, mp4_to_gif, scale_video, crop_video, accelerate_video


def test_python(args):
    print("\nRunning python tests...\n")
    test_files = args.files
    import taichi as ti
    import pytest
    if ti.is_release():
        root_dir = ti.package_root()
        test_dir = os.path.join(root_dir, 'tests')
    else:
        root_dir = ti.get_repo_directory()
        test_dir = os.path.join(root_dir, 'tests', 'python')
    pytest_args = []
    if len(test_files):
        # run individual tests
        for f in test_files:
            # auto-complete file names
            if not f.startswith('test_'):
                f = 'test_' + f
            if not f.endswith('.py'):
                f = f + '.py'
            pytest_args.append(os.path.join(test_dir, f))
    else:
        # run all the tests
        pytest_args = [test_dir]
    if args.verbose:
        pytest_args += ['-s', '-v']
    if args.rerun:
        pytest_args += ['--reruns', args.rerun]
    if int(
            pytest.main(
                [os.path.join(root_dir, 'misc/empty_pytest.py'), '-n1',
                 '-q'])) == 0:  # test if pytest has xdist or not
        try:
            from multiprocessing import cpu_count
            threads = min(8, cpu_count())  # To prevent running out of memory
        except:
            threads = 2
        os.environ['TI_DEVICE_MEMORY_GB'] = '0.5'  # Discussion: #769
        arg_threads = None
        if args.threads is not None:
            arg_threads = int(args.threads)
        env_threads = os.environ.get('TI_TEST_THREADS', '')
        if arg_threads is not None:
            threads = arg_threads
        elif env_threads:
            threads = int(env_threads)
        print(f'Starting {threads} testing thread(s)...')
        if threads > 1:
            pytest_args += ['-n', str(threads)]
    return int(pytest.main(pytest_args))


def test_cpp(args):
    import taichi as ti
    test_files = args.files
    # Cpp tests use the legacy non LLVM backend
    ti.reset()
    print("Running C++ tests...")
    task = ti.Task('test')
    return int(task.run(*test_files))


def get_benchmark_baseline_dir():
    import taichi as ti
    return os.path.join(ti.core.get_repo_dir(), 'benchmarks', 'baseline')


def get_benchmark_output_dir():
    import taichi as ti
    return os.path.join(ti.core.get_repo_dir(), 'benchmarks', 'output')


def display_benchmark_regression(xd, yd, args):
    def parse_dat(file):
        dict = {}
        for line in open(file).readlines():
            try: a, b = line.strip().split(':')
            except: continue
            dict[a.strip()] = int(float(b))
        return dict

    def parse_name(file):
        return file[5:-4].replace('__test_', '::', 1)

    def get_dats(dir):
        list = []
        for x in os.listdir(dir):
            if x.endswith('.dat'):
                list.append(x)
        dict = {}
        for x in list:
            name = parse_name(x)
            path = os.path.join(dir, x)
            dict[name] = parse_dat(path)
        return dict

    def plot_in_gui(scatter):
        import numpy as np
        import taichi as ti
        gui = ti.GUI('Regression Test', (640, 480), 0x001122)
        print('[Hint] press SPACE to go for next display')
        for key, data in scatter.items():
            data = np.array([((i + 0.5)/len(data), x/2) for i, x in enumerate(data)])
            while not gui.get_event((ti.GUI.PRESS, ti.GUI.SPACE)):
                gui.core.title = key
                gui.line((0, 0.5), (1, 0.5), 1.8, 0x66ccff)
                gui.circles(data, 0xffcc66, 1.5)
                gui.show()

    spec = args.files
    single_line = spec and len(spec) == 1
    xs, ys = get_dats(xd), get_dats(yd)
    scatter = defaultdict(list)
    for name in set(xs.keys()).union(ys.keys()):
        file, func = name.split('::')
        u, v = xs.get(name, {}), ys.get(name, {})
        ret = ''
        for key in set(u.keys()).union(v.keys()):
            if spec and key not in spec:
                continue
            a, b = u.get(key, 0), v.get(key, 0)
            res = b / a if a != 0 else math.inf
            scatter[key].append(res)
            if res == 1: continue
            if single_line:
                ret += f'{file:_<24}{func:_<42}'
            else:
                ret += f'{key:<43}'
            res -= 1
            color = Fore.RESET
            if res > 0: color = Fore.RED
            elif res < 0: color = Fore.GREEN
            ret += f'{Fore.MAGENTA}{a:>5}{Fore.RESET} -> '
            ret += f'{Fore.CYAN}{b:>5} {color}{res:>+8.1%}{Fore.RESET}\n'
        if ret != '':
            if not single_line:
                print(f'{file:_<24}{func:_<42}')
            print(ret, end='')
            if not single_line:
                print('')

    if args.gui:
        plot_in_gui(scatter)


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Run with verbose outputs')
    parser.add_argument('-r', '--rerun', help='Rerun failed tests for given times')
    parser.add_argument('-t',
                        '--threads',
                        help='Number of threads for parallel testing')
    parser.add_argument('-c',
                        '--cpp',
                        action='store_true',
                        help='Run C++ tests')
    parser.add_argument('-g',
                        '--gui',
                        action='store_true',
                        help='Display benchmark regression result in GUI')
    parser.add_argument(
        '-a',
        '--arch',
        help='Specify arch(s) to run test on, e.g. -a opengl,metal')
    parser.add_argument(
        '-n',
        '--exclusive',
        action='store_true',
        help='Exclude arch(s) instead of include, e.g. -na opengl,metal')
    parser.add_argument('files', nargs='*', help='Files to be tested')
    return parser


def main(debug=False):
    argc = len(sys.argv)
    if argc == 1:
        mode = 'help'
        parser_args = sys.argv
    else:
        mode = sys.argv[1]
        parser_args = sys.argv[2:]
    parser = make_argument_parser()
    args = parser.parse_args(args=parser_args)

    lines = []
    print()
    lines.append(u' *******************************************')
    lines.append(u' **     Taichi Programming Language       **')
    lines.append(u' *******************************************')
    if 'TI_DEBUG' in os.environ:
        val = os.environ['TI_DEBUG']
        if val not in ['0', '1']:
            raise ValueError(
                "Environment variable TI_DEBUG can only have value 0 or 1.")
    if debug:
        lines.append(u' *****************Debug Mode****************')
        os.environ['TI_DEBUG'] = '1'
    print(u'\n'.join(lines))
    print()
    import taichi as ti
    if args.arch is not None:
        arch = args.arch
        if args.exclusive:
            arch = '^' + arch
        print(f'Running on Arch={arch}')
        os.environ['TI_WANTED_ARCHS'] = arch

    if mode == 'help':
        print(
            "    Usage: ti run [task name]        |-> Run a specific task\n"
            "           ti test                   |-> Run all the tests\n"
            "           ti benchmark              |-> Run python tests in benchmark mode\n"
            "           ti baseline               |-> Archive current benchmark result as baseline\n"
            "           ti regression             |-> Display benchmark regression test result\n"
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
        return 0

    t = time.time()
    if mode.endswith('.py'):
        import subprocess
        subprocess.call([sys.executable, mode] + sys.argv[1:])
    elif mode == "run":
        if argc <= 1:
            print("Please specify [task name], e.g. test_math")
            return -1
        print(sys.argv)
        name = sys.argv[1]
        task = ti.Task(name)
        task.run(*sys.argv[2:])
    elif mode == "debug":
        ti.core.set_core_trigger_gdb_when_crash(True)
        if argc <= 2:
            print("Please specify [file name], e.g. render.py")
            return -1
        name = sys.argv[2]
        with open(name) as script:
            script = script.read()
        exec(script, {'__name__': '__main__'})
    elif mode == "test":
        if len(args.files):
            if args.cpp:
                return test_cpp(args)
            else:
                return test_python(args)
        elif args.cpp:
            return test_cpp(args)
        else:
            ret = test_python(args)
            if ret != 0:
                return ret
            return test_cpp(args)
    elif mode == "benchmark":
        import shutil
        commit_hash = ti.core.get_commit_hash()
        with os.popen('git rev-parse HEAD') as f:
            current_commit_hash = f.read().strip()
        assert commit_hash == current_commit_hash, f"Built commit {commit_hash:.6} differs from current commit {current_commit_hash:.6}, refuse to benchmark"
        os.environ['TI_PRINT_BENCHMARK_STAT'] = '1'
        output_dir = get_benchmark_output_dir()
        shutil.rmtree(output_dir, True)
        os.mkdir(output_dir)
        os.environ['TI_BENCHMARK_OUTPUT_DIR'] = output_dir
        if os.environ.get('TI_WANTED_ARCHS') is None:
            # since we only do number-of-statements benchmark
            os.environ['TI_WANTED_ARCHS'] = 'x64'
        test_python(args)
    elif mode == "baseline":
        import shutil
        baseline_dir = get_benchmark_baseline_dir()
        output_dir = get_benchmark_output_dir()
        shutil.rmtree(baseline_dir, True)
        shutil.copytree(output_dir, baseline_dir)
        print('[benchmark] baseline data saved')
    elif mode == "regression":
        baseline_dir = get_benchmark_baseline_dir()
        output_dir = get_benchmark_output_dir()
        display_benchmark_regression(baseline_dir, output_dir, args)
    elif mode == "build":
        ti.core.build()
    elif mode == "format":
        diff = None
        if len(sys.argv) >= 3:
            diff = sys.argv[2]
        ti.core.format(diff=diff)
    elif mode == "format_all":
        ti.core.format(all=True)
    elif mode == "statement":
        exec(sys.argv[2])
    elif mode == "update":
        ti.core.update(True)
        ti.core.build()
    elif mode == "asm":
        fn = sys.argv[2]
        os.system(
            r"sed '/^\s*\.\(L[A-Z]\|[a-z]\)/ d' {0} > clean_{0}".format(fn))
    elif mode == "interpolate":
        interpolate_frames('.')
    elif mode == "doc":
        os.system('cd {}/docs && sphinx-build -b html . build'.format(
            ti.get_repo_directory()))
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
            return -1
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
            return -1
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
            tmp_fn = '/tmp/{}.{:05d}.backup'.format(fn,
                                                    random.randint(0, 10000))
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
    return 0


def main_debug():
    main(debug=True)


if __name__ == '__main__':
    exit(main())
