from taichi.lang.impl import grouped, root, static
from taichi.lang.kernel_impl import kernel
from taichi.lang.misc import ij, ijk
from taichi.lang.snode import is_active
from taichi.lang.struct import Struct
from taichi.types.annotations import template
from taichi.types.primitive_types import f32


def grid(field_dict, shape):
    """Creates a 2D/3D sparse grid with each element is a struct. The struct is placed on a bitmasked snode.

    Args:
        field_dict (dict): a dict, each item is like `name: type`.
        shape (Tuple[int]): shape of the field.
    Returns:
        x: the created sparse grid, which is a bitmasked `ti.Struct.field`.

    Examples::
        # create a 2D sparse grid
        >>> grid = ti.sparse.grid({'pos': ti.math.vec2, 'mass': ti.f32, 'grid2particles': ti.types.vector(20, ti.i32)}, shape=(10, 10))

        # access
        >>> grid[0, 0].pos = ti.math.vec2(1.0, 2.0)
        >>> grid[0, 0].mass = 1.0
        >>> grid[0, 0].grid2particles[2] = 123

        # print the usage of the sparse grid, which is in [0,1]
        >>> print(ti.sparse.usage(grid))
        # 0.009999999776482582
    """
    x = Struct.field(field_dict)
    if len(shape) == 2:
        snode = root.bitmasked(ij, shape)
        snode.place(x)
    elif len(shape) == 3:
        snode = root.bitmasked(ijk, shape)
        snode.place(x)
    else:
        raise Exception("Only 2D and 3D sparse grids are supported")
    return x


@kernel
def usage(x: template()) -> f32:
    """
    Get the usage of the sparse grid, which is in [0,1]

    Args:
        x(struct field): the sparse grid to be checked.
    Returns:
        usage(f32): the usage of the sparse grid, which is in [0,1]

    Examples::
        >>> grid = ti.sparse.grid({'pos': ti.math.vec2, 'mass': ti.f32, 'grid2particles': ti.types.vector(20, ti.i32)}, shape=(10, 10))
        >>> grid[0, 0].mass = 1.0
        >>> print(ti.sparse.usage(grid))
        # 0.009999999776482582
    """
    cnt = 0
    for I in grouped(x.parent()):
        if is_active(x.parent(), I):
            cnt += 1
    total = 1.0
    if static(len(x.shape) == 2):
        total = x.shape[0] * x.shape[1]
    elif static(len(x.shape) == 3):
        total = x.shape[0] * x.shape[1] * x.shape[2]
    else:
        raise ValueError("The dimension of the sparse grid should be 2 or 3")
    res = cnt / total
    return res


__all__ = ["grid", "usage"]
