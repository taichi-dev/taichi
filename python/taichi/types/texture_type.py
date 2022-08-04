class TextureType:
    """Type annotation for Textures.

    Args:
        num_dimensions (int): Number of dimensions. For examples for a 2D texture this should be `2`.
    """
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions


class RWTextureType:
    """Type annotation for RW Textures (image load store).

    Args:
        num_dimensions (int): Number of dimensions. For examples for a 2D texture this should be `2`.
        num_channels (int): Number of channels in the texture.
        channel_format (DataType): Data type of texture
        log (float): Specifies the explicit level-of-detail.
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
