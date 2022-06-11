from taichi.lang.enums import AutodiffMode


class ForwardModeManagerImpl:
    def __init__(self, runtime, recover_kernels=True):
        self.calls = []
        self.entered = False
        self.kernels_recovered = False
        self.runtime = runtime
        self.recover_kernels_on_exit = recover_kernels

    def __enter__(self):
        self.runtime.fwd_mode_manager = self
        assert not self.entered, "Forward mode manager can be entered only once."
        self.entered = True

    def __exit__(self, _type, value, tb):
        self.runtime.fwd_mode_manager = None
        if self.recover_kernels_on_exit:
            self.recover_kernels()

    def insert(self, func):
        self.calls.append(func)

    def recover_kernels(self):
        assert self.entered, "Before recover the kernels, fwd mode manager must be entered."
        for f in self.calls:
            f.autodiff_mode = AutodiffMode.NONE
            f.compiled_functions = f.runtime.compiled_functions
        self.kernels_recovered = True
