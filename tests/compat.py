import glob
import pathlib
import os
import subprocess
import json

aot_files = glob.glob("tests/cpp/aot/python_scripts/*.py")
print(aot_files)
for x in aot_files:
    path_name = pathlib.Path(x).name[:-3]
    os.mkdir('tests/cpp/aot/python_scripts/'+path_name)
    os.environ["TAICHI_AOT_FOLDER_PATH"] = 'tests/cpp/aot/python_scripts/'+path_name
    try:
        subprocess.check_call(["python", x, "--arch=vulkan"])
    except subprocess.CalledProcessError:
        continue
    



