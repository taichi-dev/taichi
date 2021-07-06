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
from taichi.core import ti_core as _ti_core
from taichi.tools.video import (accelerate_video, crop_video, make_video,
                                mp4_to_gif, scale_video)

import taichi as ti


def timer(func):
    """Function decorator to benchmark a function runnign time."""
    import timeit

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f">>> Running time: {elapsed:.2f}s")
        return result

    return wrapper


def registerableCLI(cls):
    """Class decorator to register methodss with @register into a set."""
    cls.registered_commands = set([])
    for name in dir(cls):
        method = getattr(cls, name)
        if hasattr(method, 'registered'):
            cls.registered_commands.add(name)
    return cls


def register(func):
    """Method decorator to register CLI commands."""
    func.registered = True
    return func


@registerableCLI
class TaichiMain:
    def __init__(self, test_mode: bool = False):
        self.banner = f"\n{'*' * 43}\n**      Taichi Programming Language      **\n{'*' * 43}"
        print(self.banner)

        print(self._get_friend_links())

        parser = argparse.ArgumentParser(description="Taichi CLI",
                                         usage=self._usage())
        parser.add_argument('command',
                            help="command from the above list to run")

        # Flag for unit testing
        self.test_mode = test_mode

        self.main_parser = parser

    @timer
    def __call__(self):
        # Print help if no command provided
        if len(sys.argv[1:2]) == 0:
            self.main_parser.print_help()
            return 1

        # Parse the command
        args = self.main_parser.parse_args(sys.argv[1:2])

        if args.command not in self.registered_commands:
            # TODO: do we really need this?
            if args.command.endswith(".py"):
                TaichiMain._exec_python_file(args.command)
            else:
                print(f"{args.command} is not a valid command!")
                self.main_parser.print_help()
            return 1

        return getattr(self, args.command)(sys.argv[2:])

    def _get_friend_links(self):
        uri = 'en/stable'
        try:
            import locale
            if 'zh' in locale.getdefaultlocale()[0]:
                uri = 'zh_CN/latest'
        except:
            pass
        return '\n' \
               f'Docs:   https://taichi.rtfd.io/{uri}\n' \
               f'GitHub: https://github.com/taichi-dev/taichi\n' \
               f'Forum:  https://forum.taichi.graphics\n'

    def _usage(self) -> str:
        """Compose deterministic usage message based on registered_commands."""
        # TODO: add some color to commands
        msg = "\n"
        space = 20
        for command in sorted(self.registered_commands):
            msg += f"    {command}{' ' * (space - len(command))}|-> {getattr(self, command).__doc__}\n"
        return msg

    @staticmethod
    def _exec_python_file(filename: str):
        """Execute a Python file based on filename."""
        # TODO: do we really need this?
        import subprocess
        subprocess.call([sys.executable, filename] + sys.argv[1:])

    @staticmethod
    def _get_examples_dir() -> Path:
        """Get the path to the examples directory."""
        import taichi as ti

        root_dir = ti.package_root() if ti.is_release(
        ) else ti.get_repo_directory()
        examples_dir = Path(root_dir) / 'examples'
        return examples_dir

    @staticmethod
    def _get_available_examples() -> set:
        """Get a set of all available example names."""
        examples_dir = TaichiMain._get_examples_dir()
        all_examples = examples_dir.rglob('*.py')
        all_example_names = {
            Path(f).stem: Path(f).parent
            for f in all_examples
        }
        return all_example_names

    @staticmethod
    def _example_choices_type(choices):
        def support_choice_with_dot_py(choice):
            if choice.endswith('.py') and choice.split('.')[0] in choices:
                # try to find and remove python file extension
                return choice.split('.')[0]
            else:
                return choice

        return support_choice_with_dot_py

    @register
    def example(self, arguments: list = sys.argv[2:]):
        """Run an example by name (or name.py)"""
        choices = TaichiMain._get_available_examples()

        parser = argparse.ArgumentParser(prog='ti example',
                                         description=f"{self.example.__doc__}")
        parser.add_argument(
            "name",
            help=f"Name of an example (supports .py extension too)\n",
            type=TaichiMain._example_choices_type(choices.keys()),
            choices=sorted(choices.keys()))
        parser.add_argument(
            '-p',
            '--print',
            required=False,
            dest='print',
            action='store_true',
            help="Print example source code instead of running it")
        parser.add_argument(
            '-P',
            '--pretty-print',
            required=False,
            dest='pretty_print',
            action='store_true',
            help="Like --print, but print in a rich format with line numbers")
        parser.add_argument(
            '-s',
            '--save',
            required=False,
            dest='save',
            action='store_true',
            help="Save source code to current directory instead of running it")
        args = parser.parse_args(arguments)

        examples_dir = TaichiMain._get_examples_dir()
        target = str(
            (examples_dir / choices[args.name] / f"{args.name}.py").resolve())
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(str((examples_dir / choices[args.name]).resolve()))

        # Short circuit for testing
        if self.test_mode: return args

        if args.save:
            print(f"Saving example {args.name} to current directory...")
            shutil.copy(target, '.')
            return 0

        if args.pretty_print:
            try:
                import rich.console
                import rich.syntax
            except ImportError as e:
                print('To make -P work, please: python3 -m pip install rich')
                return 1
            # https://rich.readthedocs.io/en/latest/syntax.html
            syntax = rich.syntax.Syntax.from_path(target, line_numbers=True)
            console = rich.console.Console()
            console.print(syntax)
            return 0

        if args.print:
            with open(target) as f:
                print(f.read())
            return 0

        print(f"Running example {args.name} ...")

        runpy.run_path(target, run_name='__main__')

    @register
    def changelog(self, arguments: list = sys.argv[2:]):
        """Display changelog of current version"""
        parser = argparse.ArgumentParser(
            prog='ti changelog', description=f"{self.changelog.__doc__}")
        import taichi as ti
        if ti.is_release():
            args = parser.parse_args(arguments)
            changelog_md = os.path.join(ti.package_root(), 'CHANGELOG.md')
            with open(changelog_md) as f:
                print(f.read())
        else:
            parser.add_argument(
                'version',
                nargs='?',
                type=str,
                default='master',
                help="A version (tag) that git can use to compare diff with")
            parser.add_argument(
                '-s',
                '--save',
                action='store_true',
                help="Save changelog to CHANGELOG.md instead of print to stdout"
            )
            args = parser.parse_args(arguments)

            from . import make_changelog
            res = make_changelog.main(args.version, _ti_core.get_repo_dir())
            if args.save:
                changelog_md = os.path.join(_ti_core.get_repo_dir(),
                                            'CHANGELOG.md')
                with open(changelog_md, 'w') as f:
                    f.write(res)
            else:
                print(res)

    @register
    def release(self, arguments: list = sys.argv[2:]):
        """Make source code release"""
        parser = argparse.ArgumentParser(prog='ti release',
                                         description=f"{self.release.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        import hashlib
        import zipfile

        from git import Git
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
        commit = _ti_core.get_commit_hash()[:8]
        fn = f'taichi-src-v{ver[0]}-{ver[1]}-{ver[2]}-{commit}-{md5}.zip'
        import shutil
        shutil.move('release.zip', fn)

    @staticmethod
    def _mp4_file(name: str) -> str:
        if not name.endswith('.mp4'):
            raise argparse.ArgumentTypeError("filename must be of type .mp4")
        return name

    @register
    def gif(self, arguments: list = sys.argv[2:]):
        """Convert mp4 file to gif in the same directory"""
        parser = argparse.ArgumentParser(prog='ti gif',
                                         description=f"{self.gif.__doc__}")
        parser.add_argument('-i',
                            '--input',
                            required=True,
                            dest='input_file',
                            type=TaichiMain._mp4_file,
                            help="Path to input MP4 video file")
        parser.add_argument('-f',
                            '--framerate',
                            required=False,
                            default=24,
                            dest='framerate',
                            type=int,
                            help="Frame rate of the output GIF")
        args = parser.parse_args(arguments)

        args.output_file = str(Path(args.input_file).with_suffix('.gif'))
        ti.info(f"Converting {args.input_file} to {args.output_file}")

        # Short circuit for testing
        if self.test_mode: return args
        mp4_to_gif(args.input_file, args.output_file, args.framerate)

    @register
    def video_speed(self, arguments: list = sys.argv[2:]):
        """Speed up video in the same directory"""
        parser = argparse.ArgumentParser(
            prog='ti video_speed', description=f"{self.video_speed.__doc__}")
        parser.add_argument('-i',
                            '--input',
                            required=True,
                            dest='input_file',
                            type=TaichiMain._mp4_file,
                            help="Path to input MP4 video file")
        parser.add_argument(
            '-s',
            '--speed',
            required=True,
            dest='speed',
            type=float,
            help="Speedup factor for the output MP4 based on input. (e.g. 2.0)"
        )
        args = parser.parse_args(arguments)

        args.output_file = str(
            Path(args.input_file).with_name(
                f"{Path(args.input_file).stem}-sped{Path(args.input_file).suffix}"
            ))

        # Short circuit for testing
        if self.test_mode: return args
        accelerate_video(args.input_file, args.output_file, args.speed)

    @register
    def video_crop(self, arguments: list = sys.argv[2:]):
        """Crop video in the same directory"""
        parser = argparse.ArgumentParser(
            prog='ti video_crop', description=f"{self.video_crop.__doc__}")
        parser.add_argument('-i',
                            '--input',
                            required=True,
                            dest='input_file',
                            type=TaichiMain._mp4_file,
                            help="Path to input MP4 video file")
        parser.add_argument('--x1',
                            required=True,
                            dest='x_begin',
                            type=float,
                            help="X coordinate of the beginning crop point")
        parser.add_argument('--x2',
                            required=True,
                            dest='x_end',
                            type=float,
                            help="X coordinate of the ending crop point")
        parser.add_argument('--y1',
                            required=True,
                            dest='y_begin',
                            type=float,
                            help="Y coordinate of the beginning crop point")
        parser.add_argument('--y2',
                            required=True,
                            dest='y_end',
                            type=float,
                            help="Y coordinate of the ending crop point")
        args = parser.parse_args(arguments)

        args.output_file = str(
            Path(args.input_file).with_name(
                f"{Path(args.input_file).stem}-cropped{Path(args.input_file).suffix}"
            ))

        # Short circuit for testing
        if self.test_mode: return args
        crop_video(args.input_file, args.output_file, args.x_begin, args.x_end,
                   args.y_begin, args.y_end)

    @register
    def video_scale(self, arguments: list = sys.argv[2:]):
        """Scale video resolution in the same directory"""
        parser = argparse.ArgumentParser(
            prog='ti video_scale', description=f"{self.video_scale.__doc__}")
        parser.add_argument('-i',
                            '--input',
                            required=True,
                            dest='input_file',
                            type=TaichiMain._mp4_file,
                            help="Path to input MP4 video file")
        parser.add_argument(
            '-w',
            '--ratio-width',
            required=True,
            dest='ratio_width',
            type=float,
            help="The scaling ratio of the resolution on width")
        parser.add_argument(
            '--ratio-height',
            required=False,
            default=None,
            dest='ratio_height',
            type=float,
            help=
            "The scaling ratio of the resolution on height [default: equal to ratio-width]"
        )
        args = parser.parse_args(arguments)

        if not args.ratio_height:
            args.ratio_height = args.ratio_width
        args.output_file = str(
            Path(args.input_file).with_name(
                f"{Path(args.input_file).stem}-scaled{Path(args.input_file).suffix}"
            ))

        # Short circuit for testing
        if self.test_mode: return args
        scale_video(args.input_file, args.output_file, args.ratio_width,
                    args.ratio_height)

    @register
    def video(self, arguments: list = sys.argv[2:]):
        """Make a video using *.png files in the current directory"""
        parser = argparse.ArgumentParser(prog='ti video',
                                         description=f"{self.video.__doc__}")
        parser.add_argument("inputs", nargs='*', help="PNG file(s) as inputs")
        parser.add_argument('-o',
                            '--output',
                            required=False,
                            default=Path('./video.mp4').resolve(),
                            dest='output_file',
                            type=lambda x: Path(x).resolve(),
                            help="Path to output MP4 video file")
        parser.add_argument('-f',
                            '--framerate',
                            required=False,
                            default=24,
                            dest='framerate',
                            type=int,
                            help="Frame rate of the output MP4 video")
        parser.add_argument(
            '-c',
            '--crf',
            required=False,
            default=20,
            dest='crf',
            type=int,
            help="Constant rate factor (0-51, lower is higher quality)")
        args = parser.parse_args(arguments)

        if not args.inputs:
            args.inputs = sorted(
                str(p.resolve()) for p in Path('.').glob('*.png'))

        assert 1 <= args.crf <= 51, "The range of the CRF scale is 1–51, where 1 is almost lossless, 20 is the default, and 51 is worst quality possible."

        ti.info(f'Making video using {len(args.inputs)} png files...')
        ti.info(f'frame_rate = {args.framerate}')

        # Short circuit for testing
        if self.test_mode: return args
        make_video(args.inputs,
                   output_path=str(args.output_file),
                   crf=args.crf,
                   frame_rate=args.framerate)
        ti.info(f'Done! Output video file = {args.output_file}')

    @register
    def doc(self, arguments: list = sys.argv[2:]):
        """Build documentation"""
        parser = argparse.ArgumentParser(prog='ti doc',
                                         description=f"{self.doc.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args
        os.system(
            f'cd {ti.get_repo_directory()}/docs && sphinx-build -b html . build'
        )

    @register
    def update(self, arguments: list = sys.argv[2:]):
        """Update the Taichi codebase"""
        # TODO: Test if this still works, fix if it doesn't
        parser = argparse.ArgumentParser(prog='ti update',
                                         description=f"{self.update.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args
        _ti_core.update(True)
        _ti_core.build()

    @register
    def format(self, arguments: list = sys.argv[2:]):
        """Reformat modified source files"""
        parser = argparse.ArgumentParser(prog='ti format',
                                         description=f"{self.format.__doc__}")
        parser.add_argument(
            'diff',
            nargs='?',
            type=str,
            help="A commit hash that git can use to compare diff with")
        args = parser.parse_args(arguments)

        from .code_format import main

        # Short circuit for testing
        if self.test_mode: return args
        main(diff=args.diff)

    @register
    def format_all(self, arguments: list = sys.argv[2:]):
        """Reformat all source files"""
        parser = argparse.ArgumentParser(
            prog='ti format_all', description=f"{self.format_all.__doc__}")
        args = parser.parse_args(arguments)

        from .code_format import main

        # Short circuit for testing
        if self.test_mode: return args
        main(all=True)

    @register
    def build(self, arguments: list = sys.argv[2:]):
        """Build C++ files"""
        parser = argparse.ArgumentParser(prog='ti build',
                                         description=f"{self.build.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args
        _ti_core.build()

    @staticmethod
    def _display_benchmark_regression(xd, yd, args):
        def parse_dat(file):
            dict = {}
            with open(file) as f:
                for line in f.readlines():
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

    @staticmethod
    def _get_benchmark_baseline_dir():
        import taichi as ti
        return os.path.join(_ti_core.get_repo_dir(), 'benchmarks', 'baseline')

    @staticmethod
    def _get_benchmark_output_dir():
        import taichi as ti
        return os.path.join(_ti_core.get_repo_dir(), 'benchmarks', 'output')

    @register
    def regression(self, arguments: list = sys.argv[2:]):
        """Display benchmark regression test result"""
        parser = argparse.ArgumentParser(
            prog='ti regression', description=f"{self.regression.__doc__}")
        parser.add_argument('files',
                            nargs='*',
                            help='Test file(s) to be run for benchmarking')
        parser.add_argument('-g',
                            '--gui',
                            dest='gui',
                            action='store_true',
                            help='Display benchmark regression result in GUI')
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        baseline_dir = TaichiMain._get_benchmark_baseline_dir()
        output_dir = TaichiMain._get_benchmark_output_dir()
        TaichiMain._display_benchmark_regression(baseline_dir, output_dir,
                                                 args)

    @register
    def baseline(self, arguments: list = sys.argv[2:]):
        """Archive current benchmark result as baseline"""
        parser = argparse.ArgumentParser(
            prog='ti baseline', description=f"{self.baseline.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        import shutil
        baseline_dir = TaichiMain._get_benchmark_baseline_dir()
        output_dir = TaichiMain._get_benchmark_output_dir()
        shutil.rmtree(baseline_dir, True)
        shutil.copytree(output_dir, baseline_dir)
        print(f"[benchmark] baseline data saved to {baseline_dir}")

    @staticmethod
    def _test_python(args):
        print("\nRunning Python tests...\n")
        import pytest

        import taichi as ti
        if ti.is_release():
            root_dir = ti.package_root()
            test_dir = os.path.join(root_dir, 'tests')
        else:
            root_dir = ti.get_repo_directory()
            test_dir = os.path.join(root_dir, 'tests', 'python')
        pytest_args = []

        # TODO: use pathlib to deal with suffix and stem name manipulation
        if args.files:
            # run individual tests
            for f in args.files:
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
        try:
            if args.coverage:
                pytest_args += ['--cov-branch', '--cov=python/taichi']
            if args.cov_append:
                pytest_args += ['--cov-append']
            if args.keys:
                pytest_args += ['-k', args.keys]
            if args.marks:
                pytest_args += ['-m', args.marks]
            if args.failed_first:
                pytest_args += ['--failed-first']
            if args.fail_fast:
                pytest_args += ['--exitfirst']
        except AttributeError:
            pass

        try:
            from multiprocessing import cpu_count
            threads = min(8, cpu_count())  # To prevent running out of memory
        except NotImplementedError:
            threads = 2

        if not os.environ.get('TI_DEVICE_MEMORY_GB'):
            os.environ['TI_DEVICE_MEMORY_GB'] = '0.5'  # Discussion: #769

        env_threads = os.environ.get('TI_TEST_THREADS', '')
        threads = args.threads or env_threads or threads
        print(f'Starting {threads} testing thread(s)...')
        if int(threads) > 1:
            pytest_args += ['-n', str(threads)]
        return int(pytest.main(pytest_args))

    @staticmethod
    def _test_cpp(args):
        import taichi as ti

        # Cpp tests use the legacy non LLVM backend
        ti.reset()
        print("Running C++ tests...")
        task = ti.Task('test')
        return int(task.run(*args.files))

    @register
    def benchmark(self, arguments: list = sys.argv[2:]):
        """Run Python tests in benchmark mode"""
        parser = argparse.ArgumentParser(
            prog='ti benchmark', description=f"{self.benchmark.__doc__}")
        parser.add_argument('files', nargs='*', help='Test file(s) to be run')
        parser.add_argument('-T',
                            '--tprt',
                            dest='tprt',
                            action='store_true',
                            help='Benchmark performance in terms of run time')
        parser.add_argument('-v',
                            '--verbose',
                            dest='verbose',
                            action='store_true',
                            help='Run with verbose outputs')
        parser.add_argument('-r',
                            '--rerun',
                            required=False,
                            default=None,
                            dest='rerun',
                            type=str,
                            help='Rerun failed tests for given times')
        parser.add_argument(
            '-t',
            '--threads',
            required=False,
            default=None,
            dest='threads',
            type=str,
            help='Custom number of threads for parallel testing')
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        import shutil
        commit_hash = _ti_core.get_commit_hash()
        with os.popen('git rev-parse HEAD') as f:
            current_commit_hash = f.read().strip()
        assert commit_hash == current_commit_hash, f"Built commit {commit_hash:.6} differs from current commit {current_commit_hash:.6}, refuse to benchmark"
        os.environ['TI_PRINT_BENCHMARK_STAT'] = '1'
        output_dir = TaichiMain._get_benchmark_output_dir()
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
            TaichiMain._test_python(args)

    @register
    def test(self, arguments: list = sys.argv[2:]):
        """Run the tests"""
        parser = argparse.ArgumentParser(prog='ti test',
                                         description=f"{self.test.__doc__}")
        parser.add_argument('files',
                            nargs='*',
                            help='Test name(s) to be run, e.g. "cli"')
        parser.add_argument('-c',
                            '--cpp',
                            dest='cpp',
                            action='store_true',
                            help='Only run the C++ tests')
        parser.add_argument('-v',
                            '--verbose',
                            dest='verbose',
                            action='store_true',
                            help='Run with verbose outputs')
        parser.add_argument('-r',
                            '--rerun',
                            required=False,
                            default=None,
                            dest='rerun',
                            type=str,
                            help='Rerun failed tests for given times')
        parser.add_argument('-k',
                            '--keys',
                            required=False,
                            default=None,
                            dest='keys',
                            type=str,
                            help='Only run tests that match the keys')
        parser.add_argument('-m',
                            '--marks',
                            required=False,
                            default=None,
                            dest='marks',
                            type=str,
                            help='Only run tests with specific marks')
        parser.add_argument('-f',
                            '--failed-first',
                            required=False,
                            default=None,
                            dest='failed_first',
                            action='store_true',
                            help='Run the previously failed test first')
        parser.add_argument('-x',
                            '--fail-fast',
                            required=False,
                            default=None,
                            dest='fail_fast',
                            action='store_true',
                            help='Exit instantly on the first failed test')
        parser.add_argument('-C',
                            '--coverage',
                            required=False,
                            default=None,
                            dest='coverage',
                            action='store_true',
                            help='Run tests and record the coverage result')
        parser.add_argument(
            '-A',
            '--cov-append',
            required=False,
            default=None,
            dest='cov_append',
            action='store_true',
            help=
            'Append coverage result to existing one instead of overriding it')
        parser.add_argument(
            '-t',
            '--threads',
            required=False,
            default=None,
            dest='threads',
            type=str,
            help='Custom number of threads for parallel testing')
        parser.add_argument(
            '-a',
            '--arch',
            required=False,
            default=None,
            dest='arch',
            type=str,
            help='Custom the arch(s) (backend) to run tests on')
        parser.add_argument(
            '-n',
            '--exclusive',
            required=False,
            default=False,
            dest='exclusive',
            action='store_true',
            help=
            'Exclude arch(s) from test instead of include them, together with -a'
        )

        args = parser.parse_args(arguments)

        if args.arch:
            arch = args.arch
            if args.exclusive:
                arch = '^' + arch
            print(f'Running on Arch={arch}')
            os.environ['TI_WANTED_ARCHS'] = arch

        # Short circuit for testing
        if self.test_mode: return args

        if args.files:
            if args.cpp:
                return TaichiMain._test_cpp(args)
            else:
                return TaichiMain._test_python(args)
        elif args.cpp:
            # Only run C++ tests
            return TaichiMain._test_cpp(args)
        else:
            # Run both C++ and Python tests
            ret = TaichiMain._test_python(args)
            if ret != 0:
                return ret
            return TaichiMain._test_cpp(args)

    @register
    def run(self, arguments: list = sys.argv[2:]):
        """Run a single script"""
        parser = argparse.ArgumentParser(prog='ti run',
                                         description=f"{self.run.__doc__}")
        parser.add_argument(
            'filename',
            help='A single (Python) script to run with Taichi, e.g. render.py')
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        runpy.run_path(args.filename)

    @register
    def debug(self, arguments: list = sys.argv[2:]):
        """Debug a single script"""
        parser = argparse.ArgumentParser(prog='ti debug',
                                         description=f"{self.debug.__doc__}")
        parser.add_argument(
            'filename',
            help='A single (Python) script to run with debugger, e.g. render.py'
        )
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        _ti_core.set_core_trigger_gdb_when_crash(True)
        os.environ['TI_DEBUG'] = '1'

        runpy.run_path(args.filename, run_name='__main__')

    @register
    def task(self, arguments: list = sys.argv[2:]):
        """Run a specific task"""
        parser = argparse.ArgumentParser(prog='ti task',
                                         description=f"{self.task.__doc__}")
        parser.add_argument('taskname',
                            help='A single task name to run, e.g. test_math')
        parser.add_argument('taskargs',
                            nargs='*',
                            help='Optional task argument(s) to run with task')
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode: return args

        task = ti.Task(args.taskname)
        task.run(*args.taskargs)

    @register
    def dist(self, arguments: list = sys.argv[2:]):
        """Build package and test in release mode"""
        parser = argparse.ArgumentParser(prog='ti dist',
                                         description=f"{self.dist.__doc__}")
        parser.add_argument('mode',
                            nargs='?',
                            default='test',
                            choices=['upload', 'try_upload', 'test'],
                            help='Which mode shall we run?')
        args = parser.parse_args(arguments)

        os.chdir(os.path.join(_ti_core.get_repo_dir(), 'python'))
        sys.argv.pop(0)
        sys.argv.append(args.mode)
        runpy.run_path('build.py')

    @register
    def diagnose(self, arguments: list = sys.argv[2:]):
        """System diagnose information"""
        parser = argparse.ArgumentParser(
            prog='ti diagnose', description=f"{self.diagnose.__doc__}")
        args = parser.parse_args(arguments)

        from .diagnose import main
        main()

    @register
    def cc_compose(self, arguments: list = sys.argv[2:]):
        """Compose C backend action record into a complete C file"""
        parser = argparse.ArgumentParser(
            prog='ti cc_compose', description=f"{self.cc_compose.__doc__}")
        parser.add_argument(
            'fin_name',
            help='Action record YAML file name from C backend, e.g. program.yml'
        )
        parser.add_argument(
            'fout_name', help='The output C source file name, e.g. program.c')
        parser.add_argument(
            'hdrout_name',
            help='The output C header file name, e.g. program.h')
        parser.add_argument(
            '-e',
            '--emscripten',
            required=False,
            default=False,
            dest='emscripten',
            action='store_true',
            help='Generate output C file for Emscripten instead of raw C')
        args = parser.parse_args(arguments)

        from .cc_compose import main
        main(args.fin_name, args.fout_name, args.hdrout_name, args.emscripten)

    @register
    def repl(self, arguments: list = sys.argv[2:]):
        """Start Taichi REPL / Python shell with 'import taichi as ti'"""
        parser = argparse.ArgumentParser(prog='ti repl',
                                         description=f"{self.repl.__doc__}")
        args = parser.parse_args(arguments)

        def local_scope():
            import math
            import time

            import numpy as np

            import taichi as ti
            try:
                import IPython
                IPython.embed()
            except ImportError:
                import code
                __name__ = '__console__'
                code.interact(local=locals())

        local_scope()

    @register
    def lint(self, arguments: list = sys.argv[2:]):
        """Run pylint checker for the Python codebase of Taichi"""
        parser = argparse.ArgumentParser(prog='ti lint',
                                         description=f"{self.lint.__doc__}")
        # TODO: support arguments for lint specific files
        args = parser.parse_args(arguments)

        options = [os.path.dirname(__file__)]

        from multiprocessing import cpu_count
        threads = min(8, cpu_count())
        options += ['-j', str(threads)]

        # http://pylint.pycqa.org/en/latest/user_guide/run.html
        # TODO: support redirect output to lint.log
        import pylint.lint
        pylint.lint.Run(options)


def main():
    cli = TaichiMain()
    return cli()


if __name__ == "__main__":
    sys.exit(main())
