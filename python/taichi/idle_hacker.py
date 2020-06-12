# A dirty hack injector for python/taichi/lang/shell.py

our_code = "(lambda o,k:o.path.exists(k+'ppid_'+str(o.getpid()))and(lambda f:(f.write(f'\\n===\\n'+source),f.close()))(open(k+'source','a')))(__import__('os'),'.tmp_idle_')\n"

def main():
    import code
    import shutil

    print('Injection dest:', code.__file__)

    with open(code.__file__) as f:
        if our_code[:20] in f.read():
            print('Taichi hijacking code already exists')
            return 1

    shutil.copy(code.__file__, '.tmp_idle_backup.code.py')
    print('Backup saved in .tmp_idle_backup.code.py')

    with open(code.__file__) as f:
       ret = ''
       for x in f.readlines():
           i = x.strip().find('# Case 3')
           if i == -1:
               ret += x
           else:
               ret += x + x[:i] + our_code

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
