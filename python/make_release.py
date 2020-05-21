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
    pkg_name_dash = f'taichi-{p}'
    pkg_name_underscore = pkg_name_dash.replace('-', '_')
    package = requests.get(
        f"https://pypi.python.org/pypi/{pkg_name_dash}/json").json()
    version = '0.0.75'
    wheels = package["releases"][version]
    for wheel in wheels:
        py_ver = wheel['python_version']
        print(py_ver, wheel['url'])
        fn = download(wheel['url'])
        folder = wheel['python_version'] + '-' + fn[:-4]
        package_extracted_folder = f"release/{folder}"
        with zipfile.ZipFile(fn, 'r') as zip_ref:
            zip_ref.extractall(package_extracted_folder)
        os.remove(fn)

        pkg_ver = f"{pkg_name_underscore}-{version}"
        shutil.make_archive(
            f'release/{folder}', 'zip',
            f'release/{folder}/{pkg_ver}.data/purelib/taichi/lib')
        shutil.rmtree(package_extracted_folder)
