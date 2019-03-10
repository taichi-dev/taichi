import setuptools

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

setuptools.setup(
    name='taichi',
    packages=['taichi'],
    version='0.0.19',
    description='Computer Graphics R&D Infrastructure',
    author='Yuanming Hu',
    author_email='yuanmhu@gmail.com',
    url='https://github.com/yuanming-hu/taichi',
    keywords=['graphics', 'simulation'],
    license='MIT',
    classifiers=classifiers,)

