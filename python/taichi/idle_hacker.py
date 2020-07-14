'''
A dirty hack injector for python/taichi/lang/shell.py:IDLEInspectorWrapper
'''

import os
import taichi as ti

our_code = "__import__('taichi.idle_hacker').idle_hacker.idleipc(source)"


def get_filename(pid):
    taichi_dir = os.path.dirname(os.path.abspath(ti.__file__))
    return os.path.join(taichi_dir, '.tidle_' + str(pid))


def idleipc(source):
    with open(get_filename(os.getpid()), 'a') as f:
       f.write('\n===\n' + source)


def show_error():
    try:
        import code
        path = code.__file__
    except:
        path = '/usr/lib/python3.8/code.py'
    print('It\'s detected that you are using Python IDLE in **interactive mode**.')
    print('However, Taichi could not be fully functional due to IDLE limitation, sorry :(')
    print('Either run Taichi directly from script, or use IPython or Jupyter notebook instead.')
    print('We do care about your experience, no matter which shell you prefer to use.')
    print('So, if you would like to play with Taichi in your favorite IDLE, we may do a dirty hack:')
    print(f'Open "{path}" and add the following line to `InteractiveInterpreter.runsource`, right below `# Case 3`:')
    print('''
class InteractiveInterpreter:
    ...

    def runsource(self, source, filename="<input>", symbol="single"):
        ...

        # Case 3
        __import__('taichi.idle_hacker').idle_hacker.idleipc(source)
        self.runcode(code)
        return False

    ...

        ''')
    print('If you don\'t find where to add, we provided a script to automatically inject the code:')
    print('  sudo python3 -m taichi idle_hacker')
    print('')
    print('Then, restart IDLE and enjoy, the sky is blue and we are wizards!')




def startup_clean():
    filename = get_filename(os.getppid())
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
    filename = get_filename(os.getppid())
    try:
        with open(filename) as f:
            src = f.read()
    except FileNotFoundError as e:
        show_error()
        src = ''
    return src


def main():
    import code
    import shutil

    print('Injection dest:', code.__file__)

    with open(code.__file__) as f:
        if our_code[:20] in f.read():
            print('Taichi hijacking code already exists')
            return 1

    if not os.path.exists('.tidle_backup.code.py'):
        shutil.copy(code.__file__, '.tidle_backup.code.py')
        print('Backup saved in .tidle_backup.code.py')

    with open(code.__file__) as f:
       ret = ''
       for x in f.readlines():
           i = x.find('# Case 3')
           if i == -1:
               ret += x
           else:
               ret += x + x[:i] + our_code + '\n'

    if our_code not in ret:
        print('Error: Signature "# Case 3" not found!')
        return 2

    print('Writing back result...')

    with open(code.__file__, 'w') as f: 
        f.write(ret)

    print('Done, thank for trusting!')
    return 0


if __name__ == '__main__':
    exit(main())
