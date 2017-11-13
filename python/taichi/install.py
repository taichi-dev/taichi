import pip

required_packages = ['numpy', 'Pillow', 'scipy', 'pybind11', 'flask', 'flask_cors']


def install_package(pkg):
    pip.main(['install', '--user', pkg])


def setup():
    for pkg in required_packages:
        print("Installing package: ", pkg)
        install_package(pkg)


if __name__ == '__main__':
    setup()
