import sys, os, atexit

class ShellType:
    NATIVE = 'Python shell'
    IDLE = 'Python IDLE shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'
    IPYBASED = 'IPython Based Unknown Shell'
    SCRIPT = None

def _show_idle_error_message():
    import taichi as ti
    ver = sys.version[:3]
    if ti.get_os_name() == 'win':
        # For Windows Python IDLE: sys.executable = "../pythonw.exe"
        path = os.path.join(os.path.abspath(sys.executable), f'lib/python{ver}/code.py')
    else:
        path = f'/lib/python{ver}/code.py'
    print('It\'s detected that you are using Python IDLE as interactive shell.')
    print('However, Taichi could not be fully functional due to IDLE limitation, sorry :(')
    print('We do care about your experience, no matter which shell you prefer to use.')
    print('So, if you would like to play with Taichi in your favorite IDLE, we may do a dirty hack:')
    print(f'Open "{path}" and add the following line to `InteractiveInterpreter.runsource`, right below `# Case 3`:')
    print('''
class InteractiveInterpreter:
    ...

    def runsource(self, source, filename="<input>", symbol="single"):
        ...

        # Case 3
        with open('.tmp_idle_source', 'a') as f: f.write('\\n=====\\n' + source + '\\n')  # Add this line!
        self.runcode(code)
        return False

    ...

        ''')
    print('Then, restart IDLE and enjoy, the sky is blue and we are wizards!')

def get_shell_name():
    shell = os.environ.get('TI_SHELL_TYPE')
    if shell is not None:
        return getattr(ShellType, shell.upper())

    try:  # IPython / Jupyter?
        return 'IPython ' + get_ipython().__class__.__name__
    except:
        if hasattr(__builtins__,'__IPYTHON__'):
            return self.IPYBASED

    if 'pythonw.exe' in sys.executable:  # Windows Python IDLE?
        return ShellType.IDLE

    if os.path.basename(os.environ.get('_', '')) == 'idle':  # /usr/bin/idle?
        return ShellType.IDLE

    try:
        import psutil
        # XXX: Is psutil a hard dep of taichi? What if user doesn't install it?
        proc = psutil.Process().parent()
        if proc.name() == 'idle':  # launched from KDE win menu?
            return ShellType.IDLE
        cmdline = proc.cmdline()
        # python -m idlelib?
        if 'idlelib' in cmdline:
            return ShellType.IDLE
        # sh-bomb: /usr/bin/python /usr/bin/idle?
        if len(cmdline) >= 2 and os.path.basename(cmdline[1]) == 'idle':
            return ShellType.IDLE
    except:
        pass

    try:
        import __main__ as main
        if not hasattr(main, '__file__'):  # Interactive?
            return ShellType.NATIVE
        else:  # Python script
            return ShellType.SCRIPT

    except:
        return ShellType.SCRIPT


class ShellInspectorWrapper:
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
            self.cache = {}
            def getsource(o):
                func_name = o.__name__
                if func_name in self.cache:
                    return self.cache[func_name]
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
                # =====
                # import taichi as ti
                #
                # =====
                # @ti.kernel
                # def func():    # x.find('def ') locate to here
                #     pass
                #
                # ====
                # func()
                #
                #
                # Thanking IDLE dev :( It works anyway :)
                for x in src.split('====='):
                    x = x.strip()
                    i = x.find('def ')
                    if i == -1:
                        continue
                    name = x[i + 4:].split(':', maxsplit=1)[0]
                    name = name.split('(', maxsplit=1)[0]
                    if name.strip() == func_name:
                        self.cache[func_name] = x
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
            import IPython
            def getsourcelines(o):
                lineno = IPython.core.oinspect.find_source_lines(o)
                lines = IPython.core.oinspect.getsource(o).split('\n')
                return lines, lineno

            def getsourcefile(o):
                return '<IPython>'

            self.getsource = IPython.core.oinspect.getsource
            self.getsourcelines = getsourcelines
            self.getsourcefile = getsourcefile

        else:
            raise RuntimeError(f'Shell type "{self.name}" not supported')


oinspect = ShellInspectorWrapper()

def reset_callback():
    if oinspect.name == ShellType.IDLE:
        oinspect.cache = {}
        try:
            os.unlink('.tmp_idle_source')
        except:
            pass
        else:
            print('[Taichi] File ".tmp_idle_source" cleaned')

reset_callback()
atexit.register(reset_callback)
