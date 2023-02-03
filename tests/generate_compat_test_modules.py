"""
Generate AOT modules with Taichi nightly (which is definitely an older version
from this currently building one in development branch) for
`run_c_api_compat_test.py` to consume.
"""
import os
import subprocess

from test_utils import parse_test_configs2

if __name__ == "__main__":
    if os.environ['PLATFORM'] and "m1" in os.environ['PLATFORM']:
        print("WARNING: compatibility test is ignored on m1")
        exit(0)

    os.chdir("tests")

    BASE_DIR = "tmp/aot-compat-test/"
    os.makedirs(BASE_DIR, exist_ok=True)

    for name, cmd in parse_test_configs2().items():
        TAICHI_AOT_FOLDER_PATH = BASE_DIR + name
        os.environ['TAICHI_AOT_FOLDER_PATH'] = TAICHI_AOT_FOLDER_PATH
        os.makedirs(TAICHI_AOT_FOLDER_PATH, exist_ok=True)
        print("--", *(['python'] + cmd))
        subprocess.call(['python'] + cmd)
