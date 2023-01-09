import argparse
import glob
import json
import os
import pathlib
import subprocess

curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.dirname(curr_dir)
build_dir = os.path.join(curr_dir, 'build')
cpp_test_filename = 'taichi_cpp_tests'
capi_test_filename = 'taichi_c_api_tests'
cpp_tests_path = os.path.join(build_dir, capi_test_filename)
c_api_tests_path = os.path.join(build_dir, cpp_test_filename)

run_dict = {}


def init_dict(run_dict, aot_files):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_config_path = os.path.join(curr_dir, 'test_config.json')
    with open(test_config_path, 'r') as f:
        test_config = json.loads(f.read())

    assert ("aot_test_cases" in test_config.keys())
    assert ("capi_aot_test_cases" in test_config.keys())

    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        run_dict[path_name] = []
    for cpp_test_name, value in test_config["aot_test_cases"].items():
        if value[1] != "--arch=vulkan": continue
        test_command = []
        test_command.append(cpp_tests_path)
        test_command.append(f"--gtest_filter={cpp_test_name}")
        run_dict[value[0][3][:-3]].append(test_command)

    for cpp_test_name, value in test_config["capi_aot_test_cases"].items():
        if value[1] != "--arch=vulkan": continue
        test_command = []
        test_command.append(cpp_tests_path)
        test_command.append(f"--gtest_filter={cpp_test_name}")
        run_dict[value[0][3][:-3]].append(test_command)


def generate():
    aot_files = glob.glob("tests/cpp/aot/python_scripts/*.py")
    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        os.mkdir('tests/cpp/aot/python_scripts/' + path_name)
        os.environ[
            "TAICHI_AOT_FOLDER_PATH"] = curr_dir + '/tests/cpp/aot/python_scripts/' + path_name
        try:
            subprocess.check_call(["python", x, "--arch=vulkan"])
        except subprocess.CalledProcessError:
            continue


def run():
    aot_files = glob.glob('tests/cpp/aot/python_scripts/*.py')
    init_dict(run_dict, aot_files)
    print(run_dict)
    for x in aot_files:
        path_name = pathlib.Path(x).name[:-3]
        os.environ[
            "TAICHI_AOT_FOLDER_PATH"] = curr_dir + '/tests/cpp/aot/python_scripts/' + path_name
        if len(os.listdir('tests/cpp/aot/python_scripts/' + path_name)) == 0:
            continue
        for i in run_dict[path_name]:
            print(i)
            try:
                subprocess.check_call(i, env=os.environ.copy(), cwd=build_dir)
            except subprocess.SubprocessError:
                print(os.environ["TAICHI_AOT_FOLDER_PATH"])
                print(path_name)
                print(os.listdir(os.environ["TAICHI_AOT_FOLDER_PATH"]))
                continue
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str)
    args = parser.parse_args()

    if args.kind == 'generate':
        generate()
    elif args.kind == 'run':
        run()
