import os
import sys
import urllib.request

platform = os.environ['CI_PLATFORM']
if platform.startswith('macos'):
    platform = 'macos'
elif platform.startswith('ubuntu'):
    platform = 'linux'
elif platform.startswith('windows'):
    platform = 'msvc2019'
else:
    raise Exception(f'Bad CI_PLATFORM={platform}')

llvm_url = f'https://github.com/taichi-dev/taichi_assets/releases/download/llvm10/taichi-llvm-10.0.0-{platform}.zip'
target_dir = 'taichi-llvm'
print(f'Downloading LLVM from {llvm_url}...')
urllib.request.urlretrieve(llvm_url, "taichi-llvm.zip")
print(f'Extract zip to local dir {target_dir}...')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

retcode = os.system(f"unzip taichi-llvm.zip -d {target_dir}")
sys.exit(retcode)
