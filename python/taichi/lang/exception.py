class TaichiCompilationError(Exception):
    pass


class TaichiSyntaxError(TaichiCompilationError):
    pass


class InvalidOperationError(Exception):
    pass
