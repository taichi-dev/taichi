import os

platform = os.environ['CI_PLATFORM']
path_sep = ';' if platform.startswith('windows') else ':'

def system(x):
    ret = os.system(x)
    if ret != 0:
        exit(ret >> 8)

def add_path(x):
    path_sep = ';' if platform.startswith('windows') else ':'
    os.environ['PATH'] = x + path_sep + os.environ['PATH']

add_path(os.path.join(os.getcwd(), 'taichi-llvm', 'bin'))
add_path(os.path.join(os.getcwd(), 'bin'))
os.environ['TAICHI_REPO_DIR'] = os.getcwd()
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'python')
system('python examples/laplace.py')
system('ti test -vr2 -t2')
