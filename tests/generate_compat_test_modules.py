"""
Generate AOT modules with Taichi nightly (which is definitely an older version
from this currently building one in development branch) for
`run_c_api_compat_test.py` to consume.
"""
import glob
import os
import pathlib
import subprocess
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.dirname(curr_dir)
build_dir = os.path.join(curr_dir, "build")
cpp_test_filename = "taichi_cpp_tests"
capi_test_filename = "taichi_c_api_tests"
cpp_tests_path = os.path.join(build_dir, capi_test_filename)
c_api_tests_path = os.path.join(build_dir, cpp_test_filename)


def generate():
    aot_files = glob.glob("tests/cpp/aot/python_scripts/*.py")
    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        os.mkdir("tests/cpp/aot/python_scripts/" + path_name)
        os.environ["TAICHI_AOT_FOLDER_PATH"] = curr_dir + "/tests/cpp/aot/python_scripts/" + path_name
        try:
            subprocess.check_call([sys.executable, x, "--arch=vulkan"])
        except subprocess.CalledProcessError:
            continue


if __name__ == "__main__":
    generate()
