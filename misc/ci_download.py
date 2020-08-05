import os

platform = os.environ['CI_PLATFORM']

if platform.startswith('macos'):
    platform = 'macos'
elif platform.startswith('ubuntu'):
    platform = 'linux'
elif platform.startswith('windows'):
    platform = 'msvc2019'
else:
    raise Exception(f'Bad CI_PLATFORM={platform}')

url = f'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-${platform}.zip'
os.system(f'wget {url} --waitretry=3 --tries=5 -O taichi-llvm.zip')
