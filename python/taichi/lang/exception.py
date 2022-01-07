class TaichiCompilationError(Exception):
    pass


class TaichiSyntaxError(TaichiCompilationError, SyntaxError):
    pass


class TaichiNameError(TaichiCompilationError, NameError):
    pass


class InvalidOperationError(Exception):
    pass
