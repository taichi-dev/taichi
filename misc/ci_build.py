import os

platform = os.environ['CI_PLATFORM']
path_sep = ';' if platform.startswith('windows') else ':'

cwd = os.getcwd()
os.environ['TAICHI_REPO_DIR'] = cwd
our_path = os.path.join(cwd, 'taichi-llvm', 'bin')
os.environ['PATH'] = our_path + path_sep + os.environ['PATH']
os.environ['CXX'] = 'clang++'

exit(os.system('python misc/ci_setup.py ci') >> 8)
