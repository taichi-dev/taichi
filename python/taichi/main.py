import argparse
import math
import os
import random
import runpy
import shutil
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

from colorama import Back, Fore, Style

import taichi as ti
from taichi.tools.video import (accelerate_video, crop_video, make_video, mp4_to_gif, scale_video)


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


def get_examples_dir() -> Path:
    """Get the path to the examples directory."""
    import taichi as ti

    root_dir = ti.package_root() if ti.is_release() else ti.get_repo_directory(
    )
    examples_dir = Path(root_dir) / 'examples'
    return examples_dir


def get_available_examples() -> set:
    """Get a set of all available example names."""
    examples_dir = get_examples_dir()
    all_examples = examples_dir.rglob('*.py')
    all_example_names = {
        str(f.resolve()).split('/')[-1].split('.')[0]
        for f in all_examples
    }
    return all_example_names


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
            try:
                a, b = line.strip().split(':')
            except:
                continue
            b = float(b)
            if abs(b % 1.0) < 1e-5:  # codegen_*
                b = int(b)
            dict[a.strip()] = b
        return dict

    def parse_name(file):
        if file[0:5] == 'test_':
            return file[5:-4].replace('__test_', '::', 1)
        elif file[0:10] == 'benchmark_':
            return '::'.join(reversed(file[10:-4].split('__arch_')))
        else:
            raise Exception(f'bad benchmark file name {file}')

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
            data = np.array([((i + 0.5) / len(data), x / 2)
                             for i, x in enumerate(data)])
            while not gui.get_event((ti.GUI.PRESS, ti.GUI.SPACE)):
                gui.core.title = key
                gui.line((0, 0.5), (1, 0.5), 1.8, 0x66ccff)
                gui.circles(data, 0xffcc66, 1.5)
                gui.show()

    spec = args.files
    single_line = spec and len(spec) == 1
    xs, ys = get_dats(xd), get_dats(yd)
    scatter = defaultdict(list)
    for name in reversed(sorted(set(xs.keys()).union(ys.keys()))):
        file, func = name.split('::')
        u, v = xs.get(name, {}), ys.get(name, {})
        ret = ''
        for key in set(u.keys()).union(v.keys()):
            if spec and key not in spec:
                continue
            a, b = u.get(key, 0), v.get(key, 0)
            if a == 0:
                if b == 0:
                    res = 1.0
                else:
                    res = math.inf
            else:
                res = b / a
            scatter[key].append(res)
            if res == 1: continue
            if not single_line:
                ret += f'{key:<30}'
            res -= 1
            color = Fore.RESET
            if res > 0: color = Fore.RED
            elif res < 0: color = Fore.GREEN
            if isinstance(a, float):
                a = f'{a:>7.2}'
            else:
                a = f'{a:>7}'
            if isinstance(b, float):
                b = f'{b:>7.2}'
            else:
                b = f'{b:>7}'
            ret += f'{Fore.MAGENTA}{a}{Fore.RESET} -> '
            ret += f'{Fore.CYAN}{b} {color}{res:>+9.1%}{Fore.RESET}\n'
        if ret != '':
            print(f'{file + "::" + func:_<58}', end='')
            if not single_line:
                print('')
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
    parser.add_argument('-r',
                        '--rerun',
                        help='Rerun failed tests for given times')
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
    parser.add_argument('-T',
                        '--tprt',
                        action='store_true',
                        help='Benchmark for time performance')
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


def timer(func):
    import timeit
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f">>> Running time: {elapsed:.2f}s")
        return result
    return wrapper


class TaichiMain:
    registered_commands = {
        "example",
        "release",
        "gif",
        "video_speed",
        "video_crop",
        "video_scale",
        "video",
        "doc",
        "asm",
        "update",
        "format_all",
        "format",
        "build",
        "regression",
        "baseline",
        "benchmark",
        "test",
        "debug",
        "run"
    }

    @timer
    def __init__(self, debug=False):
        self.banner = f"\n{'*' * 43}\n**   \u262f Taichi Programming Language       **\n{'*' * 43}"
        print(self.banner)

        if 'TI_DEBUG' in os.environ:
            val = os.environ['TI_DEBUG']
            if val not in ['0', '1']:
                raise ValueError(
                    "Environment variable TI_DEBUG can only have value 0 or 1.")
        if debug:
            print(f"\n{'*' * 17} Debug Mode {'*' * 17}\n")
            os.environ['TI_DEBUG'] = '1'
        
        parser = argparse.ArgumentParser(
            description="Taichi CLI",
            usage=self._usage()
        )
        if len(sys.argv[1:2]) == 0:
            parser.print_help()
            exit(1)

        parser.add_argument('command', help="command from the above list to run")
        args = parser.parse_args(sys.argv[1:2])
        if args.command not in self.registered_commands:
            if args.command.endswith(".py"):
                self._exec_python_file(args.command)
            print(f"{args.command} is not a valid command!")
            parser.print_help()
            exit(1)
        getattr(self, args.command)(sys.argv[2:])

    def _usage(self) -> str:
        """Compose deterministic usage message based on registered_commands."""
        # TODO: add some color to commands
        msg = "\n"
        space = 20
        for command in sorted(self.registered_commands):
            msg += f"{command}{' ' * (space - len(command))}|-> {getattr(self, command).__doc__}\n"
        return msg

    def _exec_python_file(self, filename: str):
        """Execute a Python file based on filename."""
        # TODO: do we really need this?
        import subprocess
        subprocess.call([sys.executable, filename] + sys.argv[1:])

    def example(self, arguments: list = sys.argv[2:]):
        """Run an example by name"""
        parser = argparse.ArgumentParser(description=f"{self.example.__doc__}")
        parser.add_argument("name", help=f"Name of an example\n", choices=sorted(get_available_examples()))
        args = parser.parse_args(arguments)

        examples_dir = get_examples_dir()
        target = str((examples_dir / f"{args.name}.py").resolve())
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(str(examples_dir.resolve()))
        print(f"Running example {args.name} ...")
        runpy.run_path(target, run_name='__main__')
        run_example(name=args.name)

    def release(self, arguments: list = sys.argv[2:]):
        """Make source code release"""
        parser = argparse.ArgumentParser(description=f"{self.release.__doc__}")
        args = parser.parse_args(arguments)

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

    @staticmethod
    def _mp4_file(name: str) -> str:
        if not name.endswith('.mp4'):
            raise argparse.ArgumentTypeError("filename must be of type .mp4")
        return name

    def gif(self, arguments: list = sys.argv[2:]):
        """Convert mp4 file to gif in the same directory"""
        parser = argparse.ArgumentParser(description=f"{self.gif.__doc__}")
        parser.add_argument('-i', '--input', required=True, dest='input_file', type=TaichiMain._mp4_file, help="Path to input MP4 video file")
        parser.add_argument('-f', '--framerate', required=False, default=24, dest='framerate', type=int, help="Frame rate of the output GIF")
        args = parser.parse_args(arguments)

        args.output_file = Path(args.input_file).with_suffix('.gif') 
        ti.info(f"Converting {args.input_file} to {args.output_file}")
        mp4_to_gif(args.input_file, args.output_file, args.framerate)

    def video_speed(self, arguments: list = sys.argv[2:]):
        """Speed up video in the same directory"""
        parser = argparse.ArgumentParser(description=f"{self.video_speed.__doc__}")
        parser.add_argument('-i', '--input', required=True, dest='input_file', type=TaichiMain._mp4_file, help="Path to input MP4 video file")
        parser.add_argument('-s', '--speed', required=True, dest='speed', type=float, help="Speedup factor for the output MP4 based on input. (e.g. 2.0)")
        args = parser.parse_args(arguments)

        args.output_file = Path(args.input_file).with_name(f"{Path(args.input_file).stem}-sped")
        accelerate_video(args.input_file, args.output_file, args.speed)

    def video_crop(self, arguments: list = sys.argv[2:]):
        """Crop video in the same directory"""
        parser = argparse.ArgumentParser(description=f"{self.video_crop.__doc__}")
        parser.add_argument('-i', '--input', required=True, dest='input_file', type=TaichiMain._mp4_file, help="Path to input MP4 video file")
        parser.add_argument('--x1', required=True, dest='x_begin', type=float, help="X coordinate of the beginning crop point")
        parser.add_argument('--x2', required=True, dest='x_end', type=float, help="X coordinate of the ending crop point")
        parser.add_argument('--y1', required=True, dest='y_begin', type=float, help="Y coordinate of the beginning crop point")
        parser.add_argument('--y2', required=True, dest='y_end', type=float, help="Y coordinate of the ending crop point")
        args = parser.parse_args(arguments)

        args.output_file = Path(args.input_file).with_name(f"{Path(args.input_file).stem}-cropped")
        crop_video(args.input_file, args.output_file, args.x_begin, args.x_end, args.y_begin, args.y_end)

    def video_scale(self, arguments: list = sys.argv[2:]):
        """Scale video resolution in the same directory"""
        parser = argparse.ArgumentParser(description=f"{self.video_scale.__doc__}")
        parser.add_argument('-i', '--input', required=True, dest='input_file', type=TaichiMain._mp4_file, help="Path to input MP4 video file")
        parser.add_argument('-w', '--ratio-width', required=True, dest='ratio_width', type=float, help="The scaling ratio of the resolution on width")
        parser.add_argument('--ratio-height', required=False, default=None, dest='ratio_height', type=float, help="The scaling ratio of the resolution on height [default: equal to ratio-width]")
        args = parser.parse_args(arguments)

        if not args.ratio_height:
            args.ratio_height = args.ratio_width
        args.output_file = Path(args.input_file).with_name(f"{Path(args.input_file).stem}-scaled")
        scale_video(args.input_file, args.output_file, args.ratio_width, args.ratio_height)

    def video(self, arguments: list = sys.argv[2:]):
        """Make a video using *.png files in the current directory"""
        parser = argparse.ArgumentParser(description=f"{self.video.__doc__}")
        parser.add_argument("inputs", nargs='*', help="A list of png files as inputs")
        parser.add_argument('-o', '--output', required=False, default=Path('./video.mp4').resolve(), dest='output_file', type=lambda x: Path(x).resolve(), help="Path to output MP4 video file")
        parser.add_argument('-f', '--framerate', required=False, default=24, dest='framerate', type=int, help="Frame rate of the output MP4 video")
        args = parser.parse_args(arguments)
        
        if not args.inputs:
            args.inputs = [str(p.resolve()) for p in Path('.').glob('*.png')]

        ti.info(f'Making video using {len(args.inputs)} png files...')
        ti.info(f'frame_rate = {args.framerate}')

        make_video(args.inputs, output_path=str(args.output_file), frame_rate=args.framerate)
        ti.info(f'Done! Output video file = {args.output_file}')

    def doc(self, arguments: list = sys.argv[2:]):
        """Build documentation"""
        parser = argparse.ArgumentParser(description=f"{self.doc.__doc__}")
        args = parser.parse_args(arguments)

        os.system(f'cd {ti.get_repo_directory()}/docs && sphinx-build -b html . build')

    def asm(self, arguments: list = sys.argv[2:]):
        """???"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.asm.__doc__}")
        args = parser.parse_args(sys.argv[2:])

        fn = sys.argv[2]
        os.system(
            r"sed '/^\s*\.\(L[A-Z]\|[a-z]\)/ d' {0} > clean_{0}".format(fn))

    def update(self, arguments: list = sys.argv[2:]):
        """Update the Taichi codebase"""
        # TODO: Test if this still works, fix if it doesn't
        parser = argparse.ArgumentParser(description=f"{self.update.__doc__}")
        args = parser.parse_args(arguments)
        ti.core.update(True)
        ti.core.build()

    def format(self, arguments: list = sys.argv[2:]):
        """Reformat modified source files"""
        parser = argparse.ArgumentParser(description=f"{self.format.__doc__}")
        parser.add_argument('-d', '--diff', required=False, default=None, dest='diff', type=str, help="A commit hash that git can use to compare diff with")
        args = parser.parse_args(arguments)

        ti.core.format(diff=args.diff)

    def format_all(self, arguments: list = sys.argv[2:]):
        """Reformat all source files"""
        parser = argparse.ArgumentParser(description=f"{self.format_all.__doc__}")
        args = parser.parse_args(arguments)
        ti.core.format(all=True)

    def build(self, arguments: list = sys.argv[2:]):
        """Build C++ files"""
        parser = argparse.ArgumentParser(description=f"{self.build.__doc__}")
        args = parser.parse_args(arguments)
        ti.core.build()

    def regression(self, arguments: list = sys.argv[2:]):
        """Display benchmark regression test result"""
        parser = argparse.ArgumentParser(description=f"{self.regression.__doc__}")
        args = parser.parse_args(arguments)

        baseline_dir = get_benchmark_baseline_dir()
        output_dir = get_benchmark_output_dir()
        display_benchmark_regression(baseline_dir, output_dir, args)

    def baseline(self, arguments: list = sys.argv[2:]):
        """Archive current benchmark result as baseline"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.baseline.__doc__}")
        args = parser.parse_args(sys.argv[2:])

        import shutil
        baseline_dir = get_benchmark_baseline_dir()
        output_dir = get_benchmark_output_dir()
        shutil.rmtree(baseline_dir, True)
        shutil.copytree(output_dir, baseline_dir)
        print('[benchmark] baseline data saved')

    def benchmark(self, arguments: list = sys.argv[2:]):
        """Run Python tests in benchmark mode"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.benchmark.__doc__}")
        args = parser.parse_args(sys.argv[2:])

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
        if os.environ.get('TI_WANTED_ARCHS') is None and not args.tprt:
            # since we only do number-of-statements benchmark for SPRT
            os.environ['TI_WANTED_ARCHS'] = 'x64'
        if args.tprt:
            os.system('python benchmarks/run.py')
            # TODO: benchmark_python(args)
        else:
            test_python(args)

    def test(self, arguments: list = sys.argv[2:]):
        """Run the tests"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.test.__doc__}")
        args = parser.parse_args(sys.argv[2:])

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

    def debug(self, arguments: list = sys.argv[2:]):
        """Debug a script"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.debug.__doc__}")
        args = parser.parse_args(sys.argv[2:])

        ti.core.set_core_trigger_gdb_when_crash(True)
        if argc <= 2:
            print("Please specify [file name], e.g. render.py")
            return -1
        name = sys.argv[2]
        with open(name) as script:
            script = script.read()

        # FIXME: exec is a security risk here!
        exec(script, {'__name__': '__main__'})

    def run(self, arguments: list = sys.argv[2:]):
        """Run a specific task"""
        # TODO: Convert the logic to use args
        parser = argparse.ArgumentParser(description=f"{self.run.__doc__}")
        args = parser.parse_args(sys.argv[2:])

        if argc <= 1:
            print("Please specify [task name], e.g. test_math")
            return -1
        print(sys.argv)
        name = sys.argv[1]
        task = ti.Task(name)
        task.run(*sys.argv[2:])


def main():
    TaichiMain()


def main_debug():
    TaichiMain(debug=True)


if __name__ == "__main__":
    main()
