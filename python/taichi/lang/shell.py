import sys, os, atexit

class ShellType:
    NATIVE = 'Python shell'
    IDLE = 'Python IDLE shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'
    IPYBASED = 'IPython Based Shell'
    SCRIPT = None

def _show_idle_error_message():
    import code
    try:
        path = code.__file__
    except:
        path = '/usr/lib/python3.8/code.py'
    print('It\'s detected that you are using Python IDLE in **interactive mode**.')
    print('However, Taichi could not be fully functional due to IDLE limitation, sorry :(')
    print('Either run Taichi directly from script, or use IPython notebook instead.')
    print('We do care about your experience, no matter which shell you prefer to use.')
    print('So, if you would like to play with Taichi in your favorite IDLE, we may do a dirty hack:')
    print(f'Open "{path}" and add the following line to `InteractiveInterpreter.runsource`, right below `# Case 3`:')
    print('''
class InteractiveInterpreter:
    ...

    def runsource(self, source, filename="<input>", symbol="single"):
        ...

        # Case 3
        (lambda o,k:o.path.exists(k+'ppid_'+str(o.getpid()))and(lambda f:(f.write(f'\\n===\\n'+source),f.close()))(open(k+'source','a')))(__import__('os'),'.tmp_idle_') # Add this line!
        self.runcode(code)
        return False

    ...

        ''')
    print('If you don\'t find where to add, we provided a script to automatically inject the code:')
    print('  sudo python3 -m taichi idle_hacker')
    print('')
    print('Then, restart IDLE and enjoy, the sky is blue and we are wizards!')


def get_shell_name():
    """
    Detect which type of shell is using.
    Can be IPython, IDLE, Python native, or none.
    """
    shell = os.environ.get('TI_SHELL_TYPE')
    if shell is not None:
        return getattr(ShellType, shell.upper())

    try:
        import __main__ as main
        if hasattr(main, '__file__'):  # Called from a script?
            return ShellType.SCRIPT
    except:
        pass

    # Let's detect which type of interactive shell is being used.
    # As you can see, huge engineering efforts are done here just to
    # make IDLE and IPython happy, wish our user really love them :)

    try:  # IPython / Jupyter?
        return 'IPython ' + get_ipython().__class__.__name__
    except:
        # Note that we can't simply do `'IPython' in sys.modules`,
        # since it seems `torch` will import IPython on it's own too..
        if hasattr(__builtins__, '__IPYTHON__'):
            return ShellType.IPYBASED

    if os.path.basename(sys.executable) == 'pythonw.exe':  # Windows Python IDLE?
        return ShellType.IDLE

    if os.path.basename(os.environ.get('_', '')) == 'idle':  # /usr/bin/idle?
        return ShellType.IDLE

    try:
        import psutil
        # XXX: Is psutil a hard dep of taichi? What if user didn't install it?
        proc = psutil.Process().parent()
        if proc.name() == 'idle':  # launched from KDE win menu?
            return ShellType.IDLE
        cmdline = proc.cmdline()
        # launched with: python -m idlelib?
        if 'idlelib' in cmdline:
            return ShellType.IDLE
        # sh-bomb: /usr/bin/python /usr/bin/idle?
        if len(cmdline) >= 2 and os.path.basename(cmdline[1]) == 'idle':
            return ShellType.IDLE
    except:
        pass

    if 'idlelib' in sys.modules:
        return ShellType.IDLE

    try:
        if getattr(sys, 'ps1', sys.flags.interactive):
            return ShellType.NATIVE
    except:
        pass

    return ShellType.SCRIPT


class ShellInspectorWrapper:
    """
    Wrapper of the `inspect` module. When interactive shell detected,
    we will redirect getsource() calls to the corresponding inspector
    provided by / suitable for each type of shell.
    """

    def __init__(self):
        self.name = get_shell_name()

        if self.name is not None:
            print('[Taichi] Interactive shell detected:', self.name)

        if self.name is None:
            # `inspect` for "Python script"
            import inspect
            self.getsource = inspect.getsource
            self.getsourcelines = inspect.getsourcelines
            self.getsourcefile = inspect.getsourcefile

        elif self.name == ShellType.NATIVE:
            # `dill.source` for "Python native shell"
            import dill
            self.getsource = dill.source.getsource
            self.getsourcelines = dill.source.getsourcelines
            self.getsourcefile = dill.source.getsourcefile

        elif self.name == ShellType.IDLE:
            # `.tmp_idle_source` for "Python IDLE shell"
            # Thanks to IDLE's lack of support with `inspect`,
            # we have to use a dirty hack to support Taichi there.
            self.idle_cache = {}

            ppid_file = '.tmp_idle_ppid_' + str(os.getppid())
            with open(ppid_file, 'w') as f:
                import taichi as ti
                ti.info(f'[Taichi] touching {ppid_file}')
                f.write('taichi')

            def getsource(o):
                func_name = o.__name__
                if func_name in self.idle_cache:
                    return self.idle_cache[func_name]
                src = None
                try:
                    with open('.tmp_idle_source') as f:
                        src = f.read()
                except FileNotFoundError as e:
                    _show_idle_error_message()
                    raise e

                # If user added our 'hacker-code' correctly,
                # then the content of .tmp_idle_source should be:
                #
                # ===
                # import taichi as ti
                #
                # ===
                # @ti.kernel
                # def func():    # x.find('def ') locate to here
                #     pass
                #
                # ===
                # func()
                #
                #
                # Thanking IDLE dev :( It works anyway :)
                for x in src.split('==='):
                    x = x.strip()
                    i = x.find('def ')
                    if i == -1:
                        continue
                    name = x[i + 4:].split(':', maxsplit=1)[0]
                    name = name.split('(', maxsplit=1)[0]
                    if name.strip() == func_name:
                        self.idle_cache[func_name] = x
                        return x
                else:
                    raise NameError(f'Could not find source for {o.__name__}!')

            def getsourcelines(o):
                lineno = 2  # TODO: consider include lineno in .tmp_idle_source?
                lines = getsource(o).split('\n')
                return lines, lineno

            def getsourcefile(o):
                return '<IDLE>'

            self.getsource = getsource
            self.getsourcelines = getsourcelines
            self.getsourcefile = getsourcefile

        elif self.name.startswith('IPython'):
            # `IPython.core.oinspect` for "IPython advanced shell"
            def getsource(o):
                import IPython
                return IPython.core.oinspect.getsource(o)

            def getsourcelines(o):
                import IPython
                lineno = IPython.core.oinspect.find_source_lines(o)
                lines = IPython.core.oinspect.getsource(o).split('\n')
                return lines, lineno

            def getsourcefile(o):
                return '<IPython>'

            self.getsource = getsource
            self.getsourcelines = getsourcelines
            self.getsourcefile = getsourcefile

        else:
            raise RuntimeError(f'Shell type "{self.name}" not supported')


oinspect = None

def _try_clean(filename):
    try:
        os.unlink(filename)
    except:
        pass
    else:
        import taichi as ti
        ti.info('File ".tmp_idle_source" cleaned')

def reset_callback():
    global oinspect
    oinspect = ShellInspectorWrapper()

    def is_idle_oinspect():
        return isinstance(oinspect, ShellInspectorWrapper) and \
                oinspect.name == ShellType.IDLE

    if is_idle_oinspect():
        _try_clean('.tmp_idle_source')

        def exit_callback():
            if is_idle_oinspect():
                _try_clean('.tmp_idle_source')
                _try_clean('.tmp_idle_ppid_' + os.getppid())
        atexit.register(exit_callback)

reset_callback()
