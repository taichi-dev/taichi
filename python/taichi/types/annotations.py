class Template:
    """Type annotation for template kernel parameter.

    See also https://docs.taichi.graphics/lang/articles/advanced/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused
    """
    def __init__(self, tensor=None, dim=None):
        self.tensor = tensor
        self.dim = dim


template = Template
"""Alias for :class:`~taichi.types.annotations.Template`.
"""


class sparse_matrix_builder:
    pass


__all__ = ['template', 'sparse_matrix_builder']
