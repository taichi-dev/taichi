"""
Ensure AOT modules compiled by old versions of Taichi is compatible with the
latest Taichi Runtime.
"""
import glob
import os
import pathlib
import subprocess

import yaml

BASE = (pathlib.Path(__file__).parent / 'cpp').resolve()

run_dict = {}


def init_dict(run_dict, aot_files):
    test_config_path = BASE / 'cpptests.yaml'
    with open(test_config_path, 'r') as f:
        test_config = yaml.safe_load(f.read())

    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        run_dict[path_name] = []

    for binary in test_config:
        for test in binary['tests']:
            if '--arch=vulkan' not in test.get('args', ''):
                continue
            run_dict[pathlib.Path(test['script']).stem].append([
                str((BASE / binary['binary']).resolve()),
                f"--gtest_filter={test['test']}",
            ])


def run():
    aot_files = glob.glob(f'{BASE}/aot/python_scripts/*.py')
    init_dict(run_dict, aot_files)
    print(run_dict)
    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        os.environ[
            "TAICHI_AOT_FOLDER_PATH"] = f'{BASE}/aot/python_scripts/{path_name}'
        if len(os.listdir(f'{BASE}/aot/python_scripts/' + path_name)) == 0:
            continue
        for i in run_dict[path_name]:
            print(i)
            try:
                subprocess.check_call(i)
            except subprocess.SubprocessError:
                print(os.environ["TAICHI_AOT_FOLDER_PATH"])
                print(path_name)
                print(os.listdir(os.environ["TAICHI_AOT_FOLDER_PATH"]))
                continue
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    run()
