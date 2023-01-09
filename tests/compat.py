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

# aot_copy_list = [c_api_tests+'--gtest_filter=CapiTest.TestCompat*']
# graph_aot_test_list = [c_api_tests+' --gtest_filter=CapiTest.TestCompatLoadAOT']
# aot_module_test_list = [c_api_tests+' --gtest_filter=CapiTest.TestBehaviorLoadAOTModuleVulkan']
# tcm_test_list = [c_api_tests+' --gtest_filter=CapiTest.*TcmAotModule']
# kernel_aot_test1_list = [c_api_tests+' --gtest_filter=CapiTest.AotTestVulkanKernel', c_api_tests+' --gtest_filter=CapiTest.DryRunVulkanAotModule']
# dense_field_aot_test_list = [cpp_tests+' --gtest_filter=GfxAotTest.VulkanDenseField']
# kernel_aot_test2_list = [cpp_tests+' --gtest_filter=GfxAotTest.VulkanKernelTest2',cpp_tests+' --gtest_filter=CGraphAotTest.VulkanRunCGraph2']
# mpm88_graph_aot_list = [c_api_tests+' --gtest_filter=CapiTest.Mpm88TestVulkan']
# sph_aot_list = [c_api_tests+' --gtest_filter=CapiTest.SphTestVulkan']
# tcm_test_list = [c_api_tests+' --gtest_filter=CapiTest.TestLoadTcmAotModule',c_api_tests+' --gtest_filter=CapiTest.TestCreateTcmAotModule']
# texture_aot_test_list = [c_api_tests+' --gtest_filter=CapiTest.GraphTestVulkanTextureGraph']

# run_dict = {'aotcopy':aot_copy_list,'aot_module_test':aot_module_test_list,'dense_field_aot_test':dense_field_aot_test_list,
# 'graph_aot_test':graph_aot_test_list,'kernel_aot_test1':kernel_aot_test1_list,'kernel_aot_test2':kernel_aot_test2_list,
# 'mpm88_grpah_aot':mpm88_graph_aot_list,'sph_aot':sph_aot_list,'tcm_test': tcm_test_list,'texture_aot_test': texture_aot_test_list}

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


# def test():
#     templist = os.path.dirname(os.path.abspath(__file__))
#     root_opath = os.path.dirname(templist)
#     print(root_opath)
#     new_dir = os.path.join(root_opath, 'build')
#     print(os.listdir(new_dir))
#     json_path = 'C:/Users/admin/JYF_Repositories/taichi_fock/tests/cpp/aot/python_scripts/dense_field_aot_test/metadata.json'
#     with open(json_path,'r') as f:
#         temp_json = json.load(f)
#         print(temp_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str)
    args = parser.parse_args()

    if args.kind == 'generate':
        generate()
    elif args.kind == 'run':
        run()
