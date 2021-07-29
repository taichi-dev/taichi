import functools
import types
from typing import Any, Optional, Sequence, Union

from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl, snode
from taichi.lang.exception import InvalidOperationError
from taichi.misc.util import warning
from taichi.snode.snode_tree import SNodeTree

_snode_registry = _ti_core.SNodeRegistry()

_Axis = _ti_core.Axis


class FieldsBuilder:
    """A builder that constructs a SNodeTree instance.

    Example::

        x = ti.field(ti.i32)
        y = ti.field(ti.f32)
        fb = ti.FieldsBuilder()
        fb.dense(ti.ij, 8).place(x)
        fb.pointer(ti.ij, 8).dense(ti.ij, 4).place(y)

        # Afer this line, `x` and `y` are placed. No more fields can be placed
        # into `fb`.
        #
        # The tree looks like the following:
        # (implicit root)
        #  |
        #  +-- dense +-- place(x)
        #  |
        #  +-- pointer +-- dense +-- place(y)
        fb.finalize()
    """
    def __init__(self):
        self._ptr = _snode_registry.create_root()
        self._root = snode.SNode(self._ptr)
        self._finalized = False
        self._empty = True

    @classmethod
    def finalized_roots(cls):
        """Gets all the roots of the finalized SNodeTree.

        Returns:
            A list of the roots of the finalized SNodeTree.
        """
        roots_ptr = []
        size = impl.get_runtime().prog.get_snode_tree_size()
        for i in range(size):
            res = impl.get_runtime().prog.get_snode_root(i)
            roots_ptr.append(snode.SNode(res))
        return roots_ptr

    @property
    def ptr(self):
        return self._ptr

    @property
    def root(self):
        return self._root

    @property
    def empty(self):
        return self._empty

    @property
    def finalized(self):
        return self._finalized

    def dense(self, indices: Union[Sequence[_Axis], _Axis],
              dimensions: Union[Sequence[int], int]):
        """Same as :func:`taichi.lang.SNode.dense`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.dense(indices, dimensions)

    def pointer(self, indices: Union[Sequence[_Axis], _Axis],
                dimensions: Union[Sequence[int], int]):
        """Same as :func:`taichi.lang.SNode.pointer`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.pointer(indices, dimensions)

    def hash(self, indices, dimensions):
        """Same as :func:`taichi.lang.SNode.hash`"""
        raise NotImplementedError()

    def dynamic(self,
                index: Union[Sequence[_Axis], _Axis],
                dimension: Union[Sequence[int], int],
                chunk_size: Optional[int] = None):
        """Same as :func:`taichi.lang.SNode.dynamic`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.dynamic(index, dimension, chunk_size)

    def bitmasked(self, indices: Union[Sequence[_Axis], _Axis],
                  dimensions: Union[Sequence[int], int]):
        """Same as :func:`taichi.lang.SNode.bitmasked`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.bitmasked(indices, dimensions)

    def bit_struct(self, num_bits: int):
        """Same as :func:`taichi.lang.SNode.bit_struct`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.bit_struct(num_bits)

    def bit_array(self, indices: Union[Sequence[_Axis], _Axis],
                  dimensions: Union[Sequence[int], int], num_bits: int):
        """Same as :func:`taichi.lang.SNode.bit_array`"""
        self._check_not_finalized()
        self._empty = False
        return self._root.bit_array(indices, dimensions, num_bits)

    def place(self,
              *args: Any,
              offset: Optional[Union[Sequence[int], int]] = None,
              shared_exponent: bool = False):
        """Same as :func:`taichi.lang.SNode.place`"""
        self._check_not_finalized()
        self._empty = False
        self._root.place(*args, offset=offset, shared_exponent=shared_exponent)

    def lazy_grad(self):
        """Same as :func:`taichi.lang.SNode.lazy_grad`"""
        # TODO: This complicates the implementation. Figure out why we need this
        self._check_not_finalized()
        self._empty = False
        self._root.lazy_grad()

    def finalize(self, raise_warning=True):
        """Constructs the SNodeTree and finalizes this builder.

        Args:
            raise_warning (bool): Raise warning or not."""
        self._check_not_finalized()
        if self._empty and raise_warning:
            warning("Finalizing an empty FieldsBuilder!")
        self._finalized = True
        return SNodeTree(
            _ti_core.finalize_snode_tree(_snode_registry, self._ptr,
                                         impl.get_runtime().prog))

    def _check_not_finalized(self):
        if self._finalized:
            raise InvalidOperationError('FieldsBuilder finalized')
