'''
A dirty hack injector for python/taichi/lang/shell.py:IDLEInspectorWrapper
'''

import os
import functools
import taichi as ti

our_code = "__import__('taichi.idle_hacker').idle_hacker.hack(InteractiveInterpreter)"


def get_taichi_dir():
    return os.path.dirname(os.path.abspath(ti.__file__))


def get_ipc_file(pid):
    return os.path.join(get_taichi_dir(), '.tidle_' + str(pid))


def get_backup_file():
    return os.path.join(get_taichi_dir(), '.tidle_backup.code.py')


def idle_ipc_write(source):
    with open(get_ipc_file(os.getpid()), 'a') as f:
        f.write('\n===\n' + source)


def hack(InteractiveInterpreter):
    old_runsource = InteractiveInterpreter.runsource

    @functools.wraps(old_runsource)
    def new_runsource(self, source, *args, **kwargs):
        idle_ipc_write(source)
        return old_runsource(self, source, *args, **kwargs)

    InteractiveInterpreter.runsource = new_runsource


def show_error():
    try:
        import code
        path = code.__file__
    except:
        path = '/usr/lib/python3.8/code.py'

    print('''Hi! Dear Taichi user:

  It's detected that you are using Python IDLE in **interactive mode**.
  However, Taichi could not be fully functional due to IDLE limitation, sorry :(
  You could either run Taichi in IDLE file mode, or use IPython / Jupyter instead.
  But we do care about your experience, no matter which shell you prefer to use.
  So, in order to play Taichi with your favorite IDLE, we may do a dirty hack:
  Open "{path}" and append the following line to the bottom of this file:

'''
          f'    {our_code}'
          '''

If you don't find where to append, we offer a script to inject the code:
''')

    if ti.get_os_name() == 'win':
        print('  python3 -m taichi idle_hacker')
    else:
        print('  sudo python3 -m taichi idle_hacker')

    print('''
  Then, restart IDLE and enjoy, the sky is blue and we are wizards!
''')


def startup_clean():
    filename = get_ipc_file(os.getppid())
    try:
        os.unlink(filename)
    except:
        pass
    else:
        import taichi as ti
        ti.info(f'File "{filename}" cleaned')


def read_ipc_file():
    # The IDLE GUI and Taichi is running in separate process,
    # So we have to create temporary files for portable IPC :(
    filename = get_ipc_file(os.getppid())
    try:
        with open(filename) as f:
            src = f.read()
    except FileNotFoundError as e:
        show_error()
        src = ''
    return src


def main(revert=False):
    import code
    import shutil

    print('Injection dest:', code.__file__)

    backup_file = get_backup_file()

    if not revert:
        with open(code.__file__) as f:
            if our_code in f.read():
                print('ERROR: Taichi hack code already exists')
                return 1

        if not os.path.exists(backup_file):
            shutil.copy(code.__file__, backup_file)
            print(f'Backup saved in {backup_file}')

        print('Appending our hack code...')

        with open(code.__file__, 'a') as f:
            f.write('\n' + our_code)

        print('Done, thank for trusting!')

    else:
        with open(code.__file__) as f:
            if our_code not in f.read():
                print('ERROR: Taichi hack code not exist')
                return 1

        if not os.path.exists(backup_file):
            print(f'ERROR: Backup file {backup_file} not found!')
            print(
                'Sorry, please consider manually remove the code or reinstall IDLE :('
            )
            return 1

        print('Moving backup file to dest...')
        shutil.move(backup_file, code.__file__)

        print('Done, sorry for the trouble!')

    return 0


if __name__ == '__main__':
    exit(main())
