import os
import platform
import taichi as ti
import requests

package = requests.get("https://pypi.python.org/pypi/taichi-nightly/json").json()
wheels = package["releases"]['0.0.73']
for wheel in wheels:
  print(wheel['python_version'])
  print(wheel['url'])
