class Template:
    """Type annotation for template kernel parameter.
    Useful for passing parameters to kernels by reference.

    See also https://docs.taichi-lang.org/docs/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused

    Example::

        >>> a = 1
        >>>
        >>> @ti.kernel
        >>> def test():
        >>>     print(a)
        >>>
        >>> @ti.kernel
        >>> def test_template(a: ti.template()):
        >>>     print(a)
        >>>
        >>> test(a)  # will print 1
        >>> test_template(a)  # will also print 1
        >>> a = 2
        >>> test(a)  # will still print 1
        >>> test_template(a)  # will print 2
    """

    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim


template = Template
"""Alias for :class:`~taichi.types.annotations.Template`.
"""


class sparse_matrix_builder:
    pass


__all__ = ["template", "sparse_matrix_builder"]
