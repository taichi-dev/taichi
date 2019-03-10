import os

os.system('python3 -m pip install --user --upgrade twine setuptools wheel')
os.system('python3 setup.py sdist bdist_wheel')
os.system('python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose')