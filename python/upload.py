import os
import platform

os.makedirs('python/lib', exist_ok=True)
os.system('cp ../build/taichi_core.so python/lib')
os.system('cp ../external/lib/* python/lib')
os.system('rm python/lib/ffmpeg')
os.system('rm dist/*.whl')
os.system('python3 -m pip install --user --upgrade twine setuptools wheel')

if platform.system() == 'Linux':
    os.system('python3 setup.py bdist_wheel -p manylinux1_x86_64')
else:
    os.system('python3 setup.py bdist_wheel')
os.system('python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u yuanming -p $PYPI_PWD')
