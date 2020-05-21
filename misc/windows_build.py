import os
import shutil
import sys


def execute_cmd(cmd):
    print('Executing', resolve_env(cmd))
    return os.system(cmd)


def resolve_env(v):
    # replace `%`
    modified = True
    while modified:
        modified = False
        for i in range(len(v)):
            if v[i] == '%':
                for j in range(i + 1, len(v)):
                    if v[j] == '%':
                        var = v[i + 1:j]
                        v = v[:i] + os.environ[var] + v[j + 1:]
                        modified = True
                        print(v)
                        break
                break
    return v


def set_env(**kwargs):
    for k, v in kwargs.items():
        v = resolve_env(v)
        print(f"Setting {k} to '{v}'")
        os.environ[k] = v


repo_dir = "E:\\repos\\taichi"
assert len(
    sys.argv
) == 3, 'Usage: windows_build.py [python_executable] [cuda_version=10.X]'
python_executable = sys.argv[1]
cuda_version = sys.argv[2]
assert cuda_version in ["10.0", "10.1", "10.2"]
print("Python =", python_executable)
print("CUDA =", cuda_version)
set_env(PYTHON=python_executable)
set_env(TAICHI_REPO_DIR=repo_dir)
set_env(PYTHONPATH="%TAICHI_REPO_DIR%\\python")
set_env(PATH=r"%TAICHI_REPO_DIR%\bin;%PATH%")
execute_cmd("clang --version")

os.chdir(repo_dir)
build_dir = os.path.join(repo_dir, 'build')
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.mkdir(build_dir)
os.chdir(build_dir)
llvm_dir = "E:\\repos\\llvm-8.0.1\\build\\installed\\lib\\cmake\\llvm"
cuda_dir = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_version}"
execute_cmd(
    f'cmake .. -G"Visual Studio 15 2017 Win64" -DPYTHON_EXECUTABLE="%PYTHON%" -DLLVM_DIR="{llvm_dir}" -DTI_WITH_CUDA:BOOL=True -DCUDA_VERSION={cuda_version} -DCUDA_DIR="f{cuda_dir}"'
)
execute_cmd(
    r'msbuild /p:Configuration=RelWithDebInfo /p:Platform=x64 /m taichi.sln')
os.chdir(repo_dir)
execute_cmd('%PYTHON% -c "import taichi"')
execute_cmd('%PYTHON% examples/laplace.py')
execute_cmd('%PYTHON% bin/taichi test')
os.chdir(os.path.join(repo_dir, 'python'))
execute_cmd('%PYTHON% build.py try_upload')
