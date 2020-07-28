import sys, os, atexit


class ShellType:
    NATIVE = 'Python shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'
    IPYBASED = 'IPython Based Shell'
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

        else:
            raise RuntimeError(f'Shell type "{name}" not supported')

    def __init__(self):
        self.name = self.get_shell_name()
        if self.name is not None:
            print(f'[Taichi] Interactive shell detected: {self.name}')

        self.inspector = self.create_inspector(self.name)

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
        import IPython
        lineno = IPython.core.oinspect.find_source_lines(o)
        return f'<IPython:{lineno}>'


oinspect = ShellInspectorWrapper()
# TODO: also detect print according to shell type
