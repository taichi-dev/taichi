
import sys, os, atexit


class ShellType:
    NATIVE = 'Python shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'
    IPYBASED = 'IPython Based Shell'
    IDLE = 'Python IDLE shell'
    SCRIPT = None


class ShellInspectorWrapper:
"""
Wrapper of the `inspect` module. When interactive shell detected,
we will redirect getsource() calls to the corresponding inspector
provided by / suitable for each type of shell.
"""

    @staticmethod
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
        # make IDLE and IPython happy. Hope our users really love them :)

        try:  # IPython / Jupyter?
            return 'IPython ' + get_ipython().__class__.__name__
        except:
            # Note that we can't simply do `'IPython' in sys.modules`,
            # since it seems `torch` will import IPython on it's own too..
            if hasattr(__builtins__, '__IPYTHON__'):
                return ShellType.IPYBASED

        if os.path.basename(sys.executable) == 'pythonw.exe':  # Windows Python IDLE?
            return ShellType.IDLE

        if 'idlelib' in sys.modules:
            return ShellType.IDLE

        try:
            if getattr(sys, 'ps1', sys.flags.interactive):
                return ShellType.NATIVE
        except:
            pass

        return ShellType.SCRIPT


    @staticmethod
    def create_inspector(name):
        if name is None:
            # `inspect` for "Python script"
            import inspect
            return inspect

        elif name == ShellType.NATIVE:
            # `dill.source` for "Python native shell"
            import dill
            return dill.source

        elif name.startswith('IPython'):
            # `IPython.core.oinspect` for "IPython advanced shell"
            return IPythonInspectorWrapper()

        elif name == ShellType.IDLE:
            # `.tmp_idle_source` for "Python IDLE shell"
            return IDLEInspectorWrapper()

        else:
            raise RuntimeError(f'Shell type "{name}" not supported')


    def __init__(self):
        name = self.get_shell_name()
        if name is not None:
            print('[Taichi] Interactive shell detected:', name)

        self.inspector = self.create_inspector(name)


    def getsource(self, o):
        return self.inspector.getsource(o)

    def getsourcelines(self, o):
        return self.inspector.getsourcelines(o)

    def getsourcefile(self, o):
        return self.inspector.getsourcefile(o)



class IPythonInspectorWrapper:
    """`inspect` module wrapper for IPython / Jupyter notebook"""

    def __init__(self):
        pass

    def getsource(self, o):
        import IPython
        return IPython.core.oinspect.getsource(o)

    def getsourcelines(self, o):
        import IPython
        lineno = IPython.core.oinspect.find_source_lines(o)
        lines = IPython.core.oinspect.getsource(o).split('\n')
        return lines, lineno

    def getsourcefile(self, o):
        return '<IPython>'


class IDLEInspectorWrapper:
    """`inspect` module wrapper for IDLE / Blender scripting module"""

    @staticmethod
    def show_idle_error_message():
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
            (lambda o,k:o.path.exists(k+'ppid_'+str(o.getpid()))and(lambda f:(f.write(f'\\n===\\n'+source),f.close()))(open(k+'source','a')))(__import__('os'),'.tmp_idle_') # Add this line!
            self.runcode(code)
            return False

        ...

            ''')
        print('If you don\'t find where to add, we provided a script to automatically inject the code:')
        print('  sudo python3 -m taichi idle_hacker')
        print('')
        print('Then, restart IDLE and enjoy, the sky is blue and we are wizards!')

    @staticmethod
    def file_clean(filename):
        try:
            os.unlink(filename)
        except:
            pass
        else:
            import taichi as ti
            ti.info(f'File "{filename}" cleaned')

    @staticmethod
    def file_create(filename):
        with open(filename, 'w') as f:
            import taichi as ti
            f.write('[Taichi/IDLE temporary IPC file]')
        ti.info(f'File "{filename}" created')

    @staticmethod
    def file_read(filename):
        try:
            with open(filename) as f:
                src = f.read()
        except FileNotFoundError as e:
            IDLEInspectorWrapper.show_idle_error_message()
            raise e

    def __init__(self):
        # Thanks to IDLE's lack of support with `inspect`,
        # we have to use a dirty hack to support Taichi there.
        self.idle_cache = {}

        # The IDLE GUI and Taichi is running in separate process,
        # So we have to create temporary files for portable IPC :(
        self.file_create('.tmp_idle_ppid_' + str(os.getppid()))
        self.file_clean('.tmp_idle_source')

        def exit_callback():
            self.file_clean('.tmp_idle_ppid_' + os.getppid())
            self.file_clean('.tmp_idle_source')

        atexit.register(exit_callback)


    def getsource(self, o):
        if func_id in self.idle_cache:
            return self.idle_cache[func_id]

        src = self.file_read('.tmp_idle_source')

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

        func_name == o.__name__

        for x in reversed(src.split('===')):
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
            raise NameError(f'Could not find source for {func_name}!')

    def getsourcelines(self, o):
        lineno = 2  # TODO: consider include lineno in .tmp_idle_source?
        lines = self.getsource(o).split('\n')
        return lines, lineno

    def getsourcefile(self, o):
        return '<IDLE>'


oinspect = ShellInspectorWrapper()
