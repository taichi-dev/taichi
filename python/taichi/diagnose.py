def main():
    print('Taichi system diagnose:')
    print('')

    import locale
    import os
    import platform
    import subprocess
    import sys

    executable = sys.executable

    print(f'python: {sys.version}')
    print(f'system: {sys.platform}')
    print(f'executable: {executable}')
    print(f'platform: {platform.platform()}')
    print(f'architecture: {" ".join(platform.architecture())}')
    print(f'uname: {platform.uname()}')

    print(f'locale: {".".join(locale.getdefaultlocale())}')
    print(f'PATH: {os.environ.get("PATH")}')
    print(f'PYTHONPATH: {sys.path}')
    print('')

    try:
        lsb_release = subprocess.check_output(['lsb_release', '-a'])
    except Exception as e:
        print(f'`lsb_release` not available: {e}')
    else:
        print(f'{lsb_release.decode()}')

    print(f'TAICHI_REPO_DIR={os.environ.get("TAICHI_REPO_DIR", "")}')
    for k, v in os.environ.items():
        if k.startswith('TI_'):
            print(f'{k}={v}')
    print('')

    def try_print(tag, expr):
        try:
            cmd = f'import taichi as ti; print("===="); print({expr}, end="")'
            ret = subprocess.check_output([executable, '-c', cmd]).decode()
            ret = ret.split('====' + os.linesep, maxsplit=1)[1]
            print(f'{tag}: {ret}')
        except Exception as e:
            print(f'{tag}: ERROR {e}')

    print('')
    try_print('import', 'ti')
    print('')
    for arch in ['cc', 'cpu', 'metal', 'opengl', 'cuda']:
        try_print(arch, f'ti.is_arch_supported(ti.{arch})')
    print('')

    try:
        glewinfo = subprocess.check_output(['glewinfo'])
    except Exception as e:
        print(f'`glewinfo` not available: {e}')
    else:
        for line in glewinfo.decode().splitlines():
            if line.startswith('OpenGL version'):
                print(line)
                continue

            exts = [
                'GL_ARB_compute_shader', 'GL_ARB_gpu_shader_int64',
                'GL_NV_shader_atomic_float', 'GL_NV_shader_atomic_float64',
                'GL_NV_shader_atomic_int64'
            ]
            if line.split(':')[0] in exts:
                print(line)

    print('')
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'])
    except Exception as e:
        print(f'`nvidia-smi` not available: {e}')
    else:
        print(f'{nvidia_smi.decode()}')

    try:
        ti_header = subprocess.check_output(
            [executable, '-c', 'import taichi'])
    except Exception as e:
        print(f'`import taichi` failed: {e}')
    else:
        print(f'{ti_header.decode()}')

    try:
        ti_init_test = subprocess.check_output(
            [executable, '-c', 'import taichi as ti; ti.init()'])
    except Exception as e:
        print(f'`ti.init()` failed: {e}')
    else:
        print(f'{ti_init_test.decode()}')

    try:
        ti_opengl_test = subprocess.check_output(
            [executable, '-c', 'import taichi as ti; ti.init(arch=ti.opengl)'])
    except Exception as e:
        print(f'Taichi OpenGL test failed: {e}')
    else:
        print(f'{ti_opengl_test.decode()}')

    try:
        ti_cuda_test = subprocess.check_output(
            [executable, '-c', 'import taichi as ti; ti.init(arch=ti.cuda)'])
    except Exception as e:
        print(f'Taichi CUDA test failed: {e}')
    else:
        print(f'{ti_cuda_test.decode()}')

    try:
        ti_laplace = subprocess.check_output(
            [executable, '-m', 'taichi', 'example', 'minimal'])
    except Exception as e:
        print(f'`examples/laplace.py` failed: {e}')
    else:
        print(f'{ti_laplace.decode()}')

    print(
        'Consider attaching this log when maintainers ask about system information.'
    )


if __name__ == '__main__':
    main()
