import glob
import pathlib
import os
import subprocess
import json



# def parse_test_configs():
#     curr_dir = os.path.dirname(os.path.abspath(__file__))
#     test_config_path = os.path.join(curr_dir, "test_config.json")
#     with open(test_config_path, "r") as f:
#         test_config = json.loads(f.read())

#     assert ("aot_test_cases" in test_config.keys())
#     assert ("capi_aot_test_cases" in test_config.keys())

#     for cpp_test_name, value in test_config["aot_test_cases"].items():
#         test_paths = value[0]
#         test_args = value[1]
#         test_config["aot_test_cases"][cpp_test_name] = [
#             os.path.join(*test_paths), test_args
#         ]

#     for cpp_test_name, value in test_config["capi_aot_test_cases"].items():
#         test_paths = value[0]
#         test_args = value[1]
#         test_config["capi_aot_test_cases"][cpp_test_name] = [
#             os.path.join(*test_paths), test_args
#         ]

#     return test_config["aot_test_cases"], test_config["capi_aot_test_cases"]


# __aot_test_cases, __capi_aot_test_cases = parse_test_configs()



aot_files = glob.glob("tests/cpp/aot/python_scripts/*.py")
print(aot_files)
for x in aot_files:
    path_name = pathlib.Path(x).name[:-3]
    os.mkdir('tests/cpp/aot/python_scripts/'+path_name)
    os.environ["TAICHI_AOT_FOLDER_PATH"] = 'tests/cpp/aot/python_scripts/'+path_name
    try:
        subprocess.check_call(["python", x, "--arch=vulkan"])
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call(["python", x,"--arch=cpu"])
        except subprocess.CalledProcessError:
            try: 
                subprocess.check_call(["python", x, "--arch=cuda"])
            except subprocess.CalledProcessError:
                subprocess.check_call(["python",x,"--arch=opengl"])
    



