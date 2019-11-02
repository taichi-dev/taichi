import os
import requests
import zipfile
import shutil

projects = ['nightly', 'nightly-cuda-10-0', 'nightly-cuda-10-1']

def download(url):
  fn = url.split('/')[-1]
  with requests.get(url, stream=True) as r:
    with open(fn, 'wb') as f:
      shutil.copyfileobj(r.raw, f)
  return fn

for p in projects:
  package = requests.get(f"https://pypi.python.org/pypi/taichi-{p}/json").json()
  wheels = package["releases"]['0.0.75']
  for wheel in wheels:
    print(wheel['python_version'], wheel['url'])
    fn = download(wheel['url'])
    folder = wheel['python_version'] + '-' + fn[:-4]
    with zipfile.ZipFile(fn, 'r') as zip_ref:
      zip_ref.extractall('release/{}'.format(folder))
    os.remove(fn)

