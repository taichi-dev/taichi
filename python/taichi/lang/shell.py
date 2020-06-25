import sys, os


class ShellType:
    NATIVE = 'Python shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'
    IPYBASED = 'IPython Based Shell'
    SCRIPT = None


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


oinspect = ShellInspectorWrapper()
