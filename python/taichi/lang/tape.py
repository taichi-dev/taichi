class TapeImpl:
    def __init__(self, runtime, loss=None):
        self.calls = []
        self.entered = False
        self.gradient_evaluated = False
        self.runtime = runtime
        self.eval_on_exit = loss is not None

    def __enter__(self):
        self.runtime.target_tape = self
        assert not self.entered, "Tape can be entered only once."
        self.entered = True

    def __exit__(self, _type, value, tb):
        # print('# kernel calls', len(self.calls))
        self.runtime.target_tape = None
        if self.eval_on_exit:
            self.grad()

    def insert(self, func, args):
        self.calls.append((func, args))

    def grad(self):
        assert self.entered, "Before evaluating gradients tape must be entered."
        assert not self.gradient_evaluated, "Gradients of grad can be evaluated only once."
        for func, args in reversed(self.calls):
            func.grad(*args)
        self.gradient_evaluated = True
