import argparse
import atexit
import os
import shutil
import tempfile

import taichi as ti


def _test_python(args, default_dir="python"):
    print("\nRunning Python tests...\n")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(curr_dir, default_dir)
    pytest_args = []

    # TODO: use pathlib to deal with suffix and stem name manipulation
    if args.files:
        # run individual tests
        for f in args.files:
            # auto-complete file names
            if not f.startswith("test_"):
                f = "test_" + f
            if not (f.endswith(".py") or f.endswith(".ipynb")):
                f = f + ".py"
            file = os.path.join(test_dir, f)
            has_tests = False
            if os.path.exists(file):
                pytest_args.append(file)
                has_tests = True
            assert has_tests, f"Test {f} does not exist."
    else:
        # run all the tests
        pytest_args = [test_dir]
    pytest_args += ["--nbmake"]
    if args.verbose:
        pytest_args += ["-v"]
    if args.rerun:
        pytest_args += ["--reruns", args.rerun]
    try:
        if args.coverage:
            pytest_args += ["--cov-branch", "--cov=python/taichi"]
        if args.cov_append:
            pytest_args += ["--cov-append"]
        if args.keys:
            pytest_args += ["-k", args.keys]
        if args.marks:
            pytest_args += ["-m", args.marks]
        if args.failed_first:
            pytest_args += ["--failed-first"]
        if args.fail_fast:
            pytest_args += ["--exitfirst"]
        if args.timeout > 0:
            pytest_args += [
                "--durations=15",
                "-p",
                "pytest_hardtle",
                f"--timeout={args.timeout}",
            ]
    except AttributeError:
        pass

    try:
        from multiprocessing import cpu_count  # pylint: disable=C0415

        threads = min(8, cpu_count())  # To prevent running out of memory
    except NotImplementedError:
        threads = 2

    env_threads = os.environ.get("TI_TEST_THREADS", "")
    threads = args.threads or env_threads or threads
    print(f"Starting {threads} testing thread(s)...")
    if args.show_output:
        pytest_args += ["-s"]
        print(f"Due to how pytest-xdist is implemented, the -s option does not work with multiple thread...")
    else:
        if int(threads) > 1:
            pytest_args += ["-n", str(threads), "--dist=worksteal"]
    import pytest  # pylint: disable=C0415

    return int(pytest.main(pytest_args))


def test():
    """Run the tests"""
    parser = argparse.ArgumentParser(description=f"Run taichi python test")
    parser.add_argument("files", nargs="*", help='Test name(s) to be run, e.g. "cli"')
    parser.add_argument(
        "-c",
        "--cpp",
        dest="cpp",
        default=False,
        action="store_true",
        help="Only run the C++ tests",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show_output",
        action="store_true",
        help="Show output (do not capture)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Run with verbose outputs",
    )
    parser.add_argument(
        "-r",
        "--rerun",
        required=False,
        default=None,
        dest="rerun",
        type=str,
        help="Rerun failed tests for given times",
    )
    parser.add_argument(
        "-k",
        "--keys",
        required=False,
        default=None,
        dest="keys",
        type=str,
        help="Only run tests that match the keys",
    )
    parser.add_argument(
        "-m",
        "--marks",
        required=False,
        default=None,
        dest="marks",
        type=str,
        help="Only run tests with specific marks",
    )
    parser.add_argument(
        "-f",
        "--failed-first",
        required=False,
        default=None,
        dest="failed_first",
        action="store_true",
        help="Run the previously failed test first",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        required=False,
        default=None,
        dest="fail_fast",
        action="store_true",
        help="Exit instantly on the first failed test",
    )
    parser.add_argument(
        "-C",
        "--coverage",
        required=False,
        default=None,
        dest="coverage",
        action="store_true",
        help="Run tests and record the coverage result",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        required=False,
        default=600,
        type=int,
        dest="timeout",
        help="Per test timeout (only apply to python tests)",
    )
    parser.add_argument(
        "-A",
        "--cov-append",
        required=False,
        default=None,
        dest="cov_append",
        action="store_true",
        help="Append coverage result to existing one instead of overriding it",
    )
    parser.add_argument(
        "-t",
        "--threads",
        required=False,
        default=None,
        dest="threads",
        type=str,
        help="Custom number of threads for parallel testing",
    )
    parser.add_argument(
        "-a",
        "--arch",
        required=False,
        default=None,
        dest="arch",
        type=str,
        help="Custom the arch(s) (backend) to run tests on",
    )
    parser.add_argument(
        "-n",
        "--exclusive",
        required=False,
        default=False,
        dest="exclusive",
        action="store_true",
        help="Exclude arch(s) from test instead of include them, together with -a",
    )
    parser.add_argument(
        "--with-offline-cache",
        action="store_true",
        default=os.environ.get("TI_TEST_OFFLINE_CACHE", "0") == "1",
        dest="with_offline_cache",
        help="Run tests with offline_cache=True",
    )
    parser.add_argument(
        "--rerun-with-offline-cache",
        type=int,
        dest="rerun_with_offline_cache",
        default=1,
        help="Rerun all tests with offline_cache=True for given times, together with --with-offline-cache",
    )

    run_count = 1
    args = parser.parse_args()
    print(args)

    if args.arch:
        arch = args.arch
        if args.exclusive:
            arch = "^" + arch
        print(f"Running on Arch={arch}")
        os.environ["TI_WANTED_ARCHS"] = arch

    if args.with_offline_cache:
        run_count += args.rerun_with_offline_cache
        args.timeout *= run_count
        tmp_cache_file_path = tempfile.mkdtemp()
        os.environ["TI_OFFLINE_CACHE"] = "1"
        os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = tmp_cache_file_path
        if not os.environ.get("TI_OFFLINE_CACHE_CLEANING_POLICY"):
            os.environ["TI_OFFLINE_CACHE_CLEANING_POLICY"] = "never"

        def print_and_remove():
            def size_of_dir(dir):
                size = 0
                for root, dirs, files in os.walk(dir):
                    size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
                return size

            size = size_of_dir(tmp_cache_file_path)
            stat = {}
            countof_tic = 0
            for p in os.listdir(tmp_cache_file_path):
                subdir_path = os.path.join(tmp_cache_file_path, p)
                if os.path.isdir(subdir_path):
                    stat[p] = len(os.listdir(subdir_path))
                elif p.endswith(".tic"):
                    countof_tic += 1
            stat["*.tic"] = countof_tic
            shutil.rmtree(tmp_cache_file_path)
            print("Summary of testing the offline cache:")
            print(f"    Simple statistics: {stat}")
            print(f"    Size of cache files: {size / 1024:.2f} KB")

        atexit.register(print_and_remove)
    else:  # Default: disable offline cache
        os.environ["TI_OFFLINE_CACHE"] = "0"

    if args.cpp:
        # C++ tests are now handled by pytest too,
        # so we can use `_test_python` to run them,
        # though they are not really python tests.
        exit(_test_python(args, "cpp"))

    for _ in range(run_count):
        ret = _test_python(args)
        if ret == 5:
            # treat 'no tests collected' as success
            ret = 0
        if ret != 0:
            exit(ret)


if __name__ == "__main__":
    test()
