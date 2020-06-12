class _ShellInspectorWrapper:
    NATIVE = 'Python Native Shell'
    IPYTHON = 'IPython TerminalInteractiveShell'
    JUPYTER = 'IPython ZMQInteractiveShell'

    def __init__(self):
        self.name = None

        try:  # IPython advanced shell?
            self.name = 'IPython ' + get_ipython().__class__.__name__
        except:
            try:  # Python native shell?
                import __main__ as main
                if not hasattr(main, '__file__'):
                    self.name = self.NATIVE
            except:
                pass

        if self.name is not None:
            print('[Taichi] Interactive shell detected:', self.name)

        if self.name is None:
            # `inspect` for "Python script"
            import inspect
            self.getsource = inspect.getsource
            self.getsourcelines = inspect.getsourcelines
            self.getsourcefile = inspect.getsourcefile

        elif self.name == self.NATIVE:
            # `dill.source` for "Python native shell"
            import dill
            self.getsource = dill.source.getsource
            self.getsourcelines = dill.source.getsourcelines
            self.getsourcefile = dill.source.getsourcefile

        elif self.name.startswith('IPython'):
            # `IPython.core.oinspect` for "IPython advanced shell"
            import IPython
            self.getsource = IPython.core.oinspect.getsource

            def getsourcelines(o):
                lineno = IPython.core.oinspect.find_source_lines(o)
                lines = IPython.core.oinspect.getsource(o).split('\n')
                return lines, lineno

            self.getsourcelines = getsourcelines

            def getsourcefile(o):
                return '<IPython>'

            self.getsourcefile = getsourcefile

        else:
            raise RuntimeError(f'Interactive shell {self.name} not supported')


oinspect = _ShellInspectorWrapper()
