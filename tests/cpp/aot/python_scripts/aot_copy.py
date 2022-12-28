import os 
import shutil

src_path = './tests/cpp/aot/aot_test_module/'

assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
dst_path = str(os.environ["TAICHI_AOT_FOLDER_PATH"]+"/test")
shutil.copytree(src_path,dst_path)