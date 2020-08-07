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

llvm_url = f'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-{suffix}.zip'
print(f'Downloading LLVM from {llvm_url}...')
os.system(f'wget {llvm_url} --waitretry=3 --tries=5 -O taichi-llvm.zip')
print(f'Unzipping LLVM pre-built binary...')
os.mkdir('taichi-llvm')
os.chdir('taichi-llvm')
if platform.startswith('windows'):
    exit(os.system('7z x ../taichi-llvm.zip') >> 8)
else:
    exit(os.system('unzip ../taichi-llvm.zip') >> 8)
