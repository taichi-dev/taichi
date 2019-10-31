import os
import platform
# import taichi as ti
import requests

projects = ['nightly', 'nightly-cuda-10-0', 'nightly-cuda-10-1']

for p in projects:
  package = requests.get(f"https://pypi.python.org/pypi/taichi-{p}/json").json()
  wheels = package["releases"]['0.0.75']
  for wheel in wheels:
    print(wheel['python_version'])
    print(wheel['url'])
