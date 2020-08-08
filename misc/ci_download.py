import os

platform = os.environ['CI_PLATFORM']

def system(x):
    print(f'[ci] executing: {x}')
    ret = os.system(x)
    if ret != 0:
        print(f'[ci] process exited with {ret}')
        exit(1)

if platform.startswith('macos'):
    suffix = 'macos'
elif platform.startswith('ubuntu'):
    suffix = 'linux'
elif platform.startswith('windows'):
    suffix = 'msvc2019'
else:
    raise Exception(f'Bad CI_PLATFORM={platform}')

llvm_url = f'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-{suffix}.zip'
print(f'Downloading LLVM from {llvm_url}...')
if platform.startswith('windows'):
    system(f'curl --retry 10 --retry-delay 5 {llvm_url} -LO')
    clang_url = 'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/clang-10.0.0-win.zip'
    print(f'Downloading Clang from {clang_url}...')
    system(f'curl --retry 10 --retry-delay 5 {clang_url} -LO')
    system('dir')
else:
    system(f'wget {llvm_url} --waitretry=3 --tries=5 -O taichi-llvm.zip')
print(f'Unzipping LLVM pre-built binary...')
os.mkdir('taichi-llvm')
if platform.startswith('windows'):
    system('7z x taichi-llvm-10.0.0-msvc2019.zip -otaichi-llvm')
    print(f'Unzipping Clang pre-built binary...')
    os.mkdir('taichi-clang')
    system('7z x clang-10.0.0-win.zip -otaichi-clang')
    system('dir')
else:
    os.chdir('taichi-llvm')
    system('unzip ../taichi-llvm.zip')
