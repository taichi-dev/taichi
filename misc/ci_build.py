import os

platform = os.environ['CI_PLATFORM']

def system(x):
    print(f'[ci] executing: {x}')
    ret = os.system(x)
    if ret != 0:
        print(f'[ci] process exited with {ret}')
        exit(1)

def add_path(x):
    path_sep = ';' if platform.startswith('windows') else ':'
    os.environ['PATH'] = x + path_sep + os.environ['PATH']

add_path(os.path.join(os.getcwd(), 'taichi-llvm', 'bin'))
os.environ['TAICHI_REPO_DIR'] = os.getcwd()
if platform.startswith('windows'):
    add_path(os.path.join(os.getcwd(), 'taichi-clang', 'bin'))
os.environ['CXX'] = 'clang++'
system('python misc/ci_setup.py ci')
if platform.startswith('windows'):
    os.mkdir('build')
    os.chdir('build')
    system('dir')
    system('path')
    system('clang --version')
    system(f'cmake .. -G "MinGW Makefiles" -DLLVM_DIR="{os.getcwd()}\\taichi_llvm\\lib\\cmake\\llvm"')
    system('make -j4')
    #system(f'cmake .. -G"Visual Studio 16 2019" -A x64 -DLLVM_DIR="{os.getcwd()}\\taichi_llvm\\lib\\cmake\\llvm"')
    #system('msbuild /p:Configuration=RelWithDebInfo /p:Platform=x64 /m taichi.sln')
