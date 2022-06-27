class TextureType:
    """Type annotation for Textures.
    """
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions


texture = TextureType
"""Alias for :class:`~taichi.types.ndarray_type.TextureType`.
"""

__all__ = ['texture']
