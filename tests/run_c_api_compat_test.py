"""
Ensure AOT modules compiled by old versions of Taichi is compatible with the
latest Taichi Runtime.
"""
import os
import subprocess

from test_utils import parse_test_configs2

import taichi as ti

if __name__ == "__main__":
    if os.environ['PLATFORM'] and "m1" in os.environ['PLATFORM']:
        print("WARNING: compatibility test is ignored on m1")
        exit(0)

    os.chdir("tests")

    if not os.path.exists("../build/taichi_c_api_tests"):
        print(
            "WARNING: ignored aot test entirely because the c-api test binary is missing"
        )
        exit(0)

    BASE_DIR = "tmp/aot-compat-test/"
    os.makedirs(BASE_DIR, exist_ok=True)

    os.environ['TI_LIB_DIR'] = os.path.join(next(iter(ti.__path__)), '_lib',
                                            'runtime')

    for name, cmd in parse_test_configs2().items():
        TAICHI_AOT_FOLDER_PATH = BASE_DIR + name
        if not os.path.exists(f"{TAICHI_AOT_FOLDER_PATH}/__version__"):
            print(
                f"WARNING: ignored aot test '{name}' because corresponding aot module is not generated"
            )
            continue
        os.environ['TAICHI_AOT_FOLDER_PATH'] = TAICHI_AOT_FOLDER_PATH

        cmd = ['../build/taichi_c_api_tests', f"--gtest_filter={name}"
               ] + cmd[1].split(' ')
        print("--", ' '.join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(e)
            exit(1)
