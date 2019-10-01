import setuptools
import glob

classifiers = [
    'Development Status :: 1 - Planning',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: C++',
]

data_files = glob.glob('python/lib/*')
print(data_files)
packages = setuptools.find_packages()
print(packages)

def is_nightly():
    return True

def project_name():
    if is_nightly():
        return 'taichi-nightly'
    else:
        return 'taichi'

setuptools.setup(
    name=project_name(),
    packages=packages,
    version='0.0.32',
    description='The Taichi Programming Language',
    author='Yuanming Hu',
    author_email='yuanmhu@gmail.com',
    url='https://github.com/yuanming-hu/taichi',
    install_requires=[
        'numpy', 'Pillow', 'scipy', 'pybind11', 'GitPython', 'yapf', 'colorama', 'psutil'
    ],
    data_files=[('lib', data_files)],
    keywords=['graphics', 'simulation'],
    license='MIT',
    platforms=['Linux'],
    include_package_data=True,
    classifiers=classifiers,
    has_ext_modules=lambda: True
)
