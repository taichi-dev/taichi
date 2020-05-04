import setuptools
import glob

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Software Development :: Compilers',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

data_files = glob.glob('python/lib/*')
print(data_files)
packages = setuptools.find_packages()
print(packages)

setuptools.setup(
    name=project_name,
    packages=packages,
    version=version,
    description='The Taichi Programming Language',
    author='Yuanming Hu',
    author_email='yuanmhu@gmail.com',
    url='https://github.com/taichi-dev/taichi',
    install_requires=[
        'numpy',
        'pybind11',
        'colorama',
        'setuptools',
        'astor',
        # For testing:
        'pytest',
    ],
    data_files=[('lib', data_files)],
    keywords=['graphics', 'simulation'],
    license='MIT',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ti=taichi.main:main',
            'tid=taichi.main:main_debug',
        ],
    },
    classifiers=classifiers,
    has_ext_modules=lambda: True)

# Note: this is a template setup.py used by python/build.py