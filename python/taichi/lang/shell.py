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
    def get_shell_name(exclude_script=False):
        """
        Detect which type of shell is using.
        Can be IPython, IDLE, Python native, or none.
        """
        shell = os.environ.get('TI_SHELL_TYPE')
        if shell is not None:
            return getattr(ShellType, shell.upper())

        if not exclude_script:
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
            try:
                import dill
            except ImportError as e:
                raise ImportError(
                    'In order to run Taichi in Python interactive shell, '
                    'Please execute `python3 -m pip install --user dill`')
            return dill.source

        elif name.startswith('IPython'):
            # `IPython.core.oinspect` for "IPython advanced shell"
            return IPythonInspectorWrapper()

        elif name == ShellType.IDLE:
            # `.tidle_xxx` for "Python IDLE shell"
            return IDLEInspectorWrapper()

        else:
            raise RuntimeError(f'Shell type "{name}" not supported')

    def __init__(self):
        self.name = self.get_shell_name()
        if self.name is not None:
            print(f'[Taichi] Interactive shell detected: {self.name}')

        self.inspector = self.create_inspector(self.name)

        if hasattr(self.inspector, 'startup_clean'):
            self.inspector.startup_clean()

    def try_reset_shell_type(self):
        new_name = self.get_shell_name(exclude_script=True)
        if self.name != new_name:
            print(
                f'[Taichi] Shell type changed from "{self.name}" to "{new_name}"'
            )

            self.name = new_name
            self.inspector = self.create_inspector(self.name)

    def _catch_forward(foo):
        """
        If Taichi starts within IDLE file mode, and after that user moved to interactive mode,
        then there will be an OSError, since it's switched from None to Python IDLE shell...
        We have to reset the shell type and create a corresponding inspector at this moment.
        """
        import functools

        @functools.wraps(foo)
        def wrapped(self, *args, **kwargs):
            try:
                return foo(self, *args, **kwargs)
            except OSError:
                self.try_reset_shell_type()
                return foo(self, *args, **kwargs)

        return wrapped

    @_catch_forward
    def getsource(self, o):
        return self.inspector.getsource(o)

    @_catch_forward
    def getsourcelines(self, o):
        return self.inspector.getsourcelines(o)

    @_catch_forward
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
        lineno = IPython.core.oinspect.find_source_lines(o)
        return f'<IPython:{lineno}>'


class IDLEInspectorWrapper:
    """`inspect` module wrapper for IDLE / Blender scripting module"""

    # Thanks to IDLE's lack of support with `inspect`,
    # we have to use a dirty hack to support Taichi there.

    def __init__(self):
        self.idle_cache = {}
        from taichi.idle_hacker import startup_clean
        self.startup_clean = startup_clean

    def getsource(self, o):
        func_id = id(o)
        if func_id in self.idle_cache:
            return self.idle_cache[func_id]

        from taichi.idle_hacker import read_ipc_file
        src = read_ipc_file()

        # If user added our 'hacker-code' correctly,
        # then the content of `.tidle_xxx` should be:
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

        func_name = o.__name__
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
        lineno = 2  # TODO: consider include lineno in .tidle_xxx?
        lines = self.getsource(o).split('\n')
        return lines, lineno

    def getsourcefile(self, o):
        return '<IDLE>'


oinspect = ShellInspectorWrapper()
# TODO: also detect print according to shell type
