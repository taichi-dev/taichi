# A dirty hack injector for python/taichi/lang/shell.py

our_code = "__import__('taichi.idle_hacker').idle_hacker.idleipc(source)"

def idleipc(source):
    import os
    import taichi as ti
    taichi_dir = os.path.dirname(os.path.abspath(ti.__file__))
    filename = os.path.join(taichi_dir, '.tidle_' + str(os.getpid()))
    with open(filename, 'a') as f:
       f.write('\n===\n' + source)

def main():
    import code
    import shutil
    import os

    print('Injection dest:', code.__file__)

    with open(code.__file__) as f:
        if our_code[:20] in f.read():
            print('Taichi hijacking code already exists')
            return 1

    if not os.path.exists('.tidle_backup.code.py'):
        shutil.copy(code.__file__, '.tidle_backup.code.py')
        print('Backup saved in .tmp_idle_backup.code.py')

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
