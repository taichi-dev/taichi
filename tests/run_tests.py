import argparse
import os
import pdb
import subprocess
import sys
import warnings

import taichi as ti


def _test_cpp():
    ti.reset()
    print("Running C++ tests...")
    ti_lib_dir = os.path.join(ti.__path__[0], '_lib', 'runtime')

    cpp_test_filename = 'taichi_cpp_tests'
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(curr_dir, '../build')
    if os.path.exists(os.path.join(build_dir, cpp_test_filename)):
        env_used_by_taichi_core = [
            'HOME',
            'XDG_CACHE_HOME',
            'PYTEST_CURRENT_TEST',
            'TI_BENCHMARK_OUTPUT_DIR',
            'TI_ENABLE_CUDA',
            'TI_ENABLE_METAL',
            'TI_USE_METAL_SIMDGROUP',
            'TI_ENABLE_OPENGL',
        ]
        env_used_by_taichi_core = {
            e: os.getenv(e)
            for e in env_used_by_taichi_core
        }
        env_used_by_taichi_core = {
            k: v
            for k, v in env_used_by_taichi_core.items() if v is not None
        }
        subprocess.check_call(f'./{cpp_test_filename}',
                              env={
                                  'TI_LIB_DIR': ti_lib_dir,
                                  **env_used_by_taichi_core
                              },
                              cwd=build_dir)
    else:
        warnings.warn(
            f"C++ tests are skipped due to missing {cpp_test_filename} in {build_dir}."
            "Try building taichi with `TAICHI_CMAKE_ARGS=\'-DTI_BUILD_TESTS:BOOL=ON\' python setup.py develop`"
            "if you want to enable it.")


def _test_python(args):
    print("\nRunning Python tests...\n")

    test_38 = sys.version_info >= (3, 8)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(curr_dir, 'python')
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
            file = os.path.join(test_dir, f)
            has_tests = False
            if os.path.exists(file):
                pytest_args.append(file)
                has_tests = True
            assert has_tests, f"Test {f} does not exist."
    else:
        # run all the tests
        pytest_args = [test_dir]
    if args.verbose:
        pytest_args += ['-v']
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
        from multiprocessing import cpu_count  # pylint: disable=C0415
        threads = min(8, cpu_count())  # To prevent running out of memory
    except NotImplementedError:
        threads = 2

    if not os.environ.get('TI_DEVICE_MEMORY_GB'):
        os.environ['TI_DEVICE_MEMORY_GB'] = '1.0'  # Discussion: #769

    env_threads = os.environ.get('TI_TEST_THREADS', '')
    threads = args.threads or env_threads or threads
    print(f'Starting {threads} testing thread(s)...')
    if args.show_output:
        pytest_args += ['-s']
        print(
            f'Due to how pytest-xdist is implemented, the -s option does not work with multiple thread...'
        )
    else:
        if int(threads) > 1:
            pytest_args += ['-n', str(threads)]
    import pytest  # pylint: disable=C0415
    return int(pytest.main(pytest_args))


def test():
    """Run the tests"""
    parser = argparse.ArgumentParser(
        description=f"Run taichi cpp & python tess")
    parser.add_argument('files',
                        nargs='*',
                        help='Test name(s) to be run, e.g. "cli"')
    parser.add_argument('-c',
                        '--cpp',
                        dest='cpp',
                        default=True,
                        action='store_true',
                        help='Run the C++ tests')
    parser.add_argument('-s',
                        '--show',
                        dest='show_output',
                        action='store_true',
                        help='Show output (do not capture)')
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
        help='Append coverage result to existing one instead of overriding it')
    parser.add_argument('-t',
                        '--threads',
                        required=False,
                        default=None,
                        dest='threads',
                        type=str,
                        help='Custom number of threads for parallel testing')
    parser.add_argument('-a',
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
        'Exclude arch(s) from test instead of include them, together with -a')

    args = parser.parse_args()
    print(args)

    if args.arch:
        arch = args.arch
        if args.exclusive:
            arch = '^' + arch
        print(f'Running on Arch={arch}')
        os.environ['TI_WANTED_ARCHS'] = arch

    if args.cpp:
        _test_cpp()

    if _test_python(args) != 0:
        exit(1)


if __name__ == '__main__':
    test()
