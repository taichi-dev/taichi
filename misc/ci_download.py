import os

platform = os.environ['CI_PLATFORM']
if platform.startswith('macos'):
    suffix = 'macos'
elif platform.startswith('ubuntu'):
    suffix = 'linux'
elif platform.startswith('windows'):
    suffix = 'msvc2019'
else:
    raise Exception(f'Bad CI_PLATFORM={platform}')

def system(x):
    ret = os.system(x)
    if ret != 0:
        exit(ret >> 8)

llvm_url = f'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-{suffix}.zip'
print(f'Downloading LLVM from {llvm_url}...')
system(f'wget {llvm_url} --waitretry=3 --tries=5 -O taichi-llvm.zip')
system(f'curl --retry 10 --retry-delay 5 ${url} -LO -o taichi-llvm.zip')
print(f'Unzipping LLVM pre-built binary...')
os.mkdir('taichi-llvm')
os.chdir('taichi-llvm')
if platform.startswith('windows'):
    system('7z x ../taichi-llvm.zip')
else:
    system('unzip ../taichi-llvm.zip')
