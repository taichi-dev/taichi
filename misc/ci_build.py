import os

platform = os.environ['CI_PLATFORM']

def system(x):
    ret = os.system(x)
    if ret != 0:
        exit(ret >> 8)

def add_path(x):
    path_sep = ';' if platform.startswith('windows') else ':'
    os.environ['PATH'] = x + path_sep + os.environ['PATH']

add_path(os.path.join(os.getcwd(), 'taichi-llvm', 'bin'))
os.environ['TAICHI_REPO_DIR'] = os.getcwd()
os.environ['CXX'] = 'clang++'
system('python misc/ci_setup.py ci')
