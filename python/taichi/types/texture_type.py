class TextureType:
    """Type annotation for Textures.
    """
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions


class RWTextureType:
    """Type annotation for RW Textures (image load store).
    """
    def __init__(self, num_dimensions, num_channels, channel_format, lod):
        self.num_dimensions = num_dimensions
        self.num_channels = num_channels
        self.channel_format = channel_format
        self.lod = lod


texture = TextureType
rw_texture = RWTextureType
"""Alias for :class:`~taichi.types.ndarray_type.TextureType`.
"""

__all__ = ['texture', 'rw_texture']
