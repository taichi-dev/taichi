class TaichiSyntaxError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class InvalidOperationError(Exception):
    pass
