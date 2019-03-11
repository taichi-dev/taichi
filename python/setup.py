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

data_files = glob.glob('lib/*')
print(data_files)
packages = setuptools.find_packages()
print(packages)

setuptools.setup(
    name='taichi',
    packages=packages,
    version='0.0.27',
    description='Computer Graphics R&D Infrastructure',
    author='Yuanming Hu',
    author_email='yuanmhu@gmail.com',
    url='https://github.com/yuanming-hu/taichi',
    install_requires=[
        'numpy', 'Pillow', 'scipy', 'pybind11', 'flask', 'flask_cors',
        'GitPython', 'yapf', 'colorama', 'psutil', 'requests', 'PyQt5'
    ],
    data_files=[('lib', data_files)],
    keywords=['graphics', 'simulation'],
    license='MIT',
    platforms=['Linux'],
    include_package_data=True,
    classifiers=classifiers,
    has_ext_modules=lambda: True
)
