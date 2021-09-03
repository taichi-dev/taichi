from enum import Enum
from typing import OrderedDict

import numpy as np
import taichi.core.primitive_types as primitive_type
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.exception import InvalidOperationError, TaichiSyntaxError
from taichi.lang.field import Field, ScalarField
from taichi.lang.kernel_arguments import template
from taichi.lang.kernel_impl import kernel
from taichi.lang.matrix import Matrix, MatrixField
from taichi.lang.snode import SNode
from taichi.lang.struct import Struct
from taichi.lang.types import CompoundType
from taichi.lang.util import (cook_dtype, has_pytorch, is_taichi_class,
                              python_scope, taichi_scope, to_pytorch_type)
from taichi.misc.mesh_loader import *

MeshTopology = _ti_core.MeshTopology
MeshElementType = _ti_core.MeshElementType
MeshRelationType = _ti_core.MeshRelationType
MeshElementReorderingType = _ti_core.MeshElementReorderingType
ConvType = _ti_core.ConvType
element_order = _ti_core.element_order
from_end_element_order = _ti_core.from_end_element_order
to_end_element_order = _ti_core.to_end_element_order
relation_by_orders = _ti_core.relation_by_orders
inverse_relation = _ti_core.inverse_relation


class MeshAttrType:
    def __init__(self, name, dtype, reordering, needs_grad):
        self.name = name
        self.dtype = dtype
        self.reordering = reordering
        self.needs_grad = needs_grad


class MeshElementField:
    def __init__(self, type, attr_dict, field_dict):
        self.type = type
        self.attr_dist = attr_dict
        self.field_dict = field_dict

        self.register_fields()

    @property
    def keys(self):
        return list(self.field_dict.keys())

    @property
    def members(self):
        return list(self.field_dict.values())

    @property
    def items(self):
        return self.field_dict.items()

    @staticmethod
    def make_getter(key):
        def getter(self):
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self.field_dict[key]

        return getter

    @staticmethod
    def make_setter(key):
        @python_scope
        def setter(self, value):
            _taichi_skip_traceback = 1
            self.field_dict[key] = value

        return setter

    def register_fields(self):
        for k in self.keys:
            setattr(
                MeshElementField, k,
                property(
                    MeshElementField.make_getter(k),
                    MeshElementField.make_setter(k),
                ))

    def get_field_members(self):
        field_members = []
        for m in self.members:
            assert isinstance(m, Field)
            field_members += m.get_field_members()
        return field_members

    @property
    def snode(self):
        return self.members[0].snode

    def loop_range(self):
        return self.members[0].loop_range()

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, Field)
        assert set(self.keys) == set(other.keys)
        for k in self.keys:
            self.field_dict[k].copy_from(other[k])

    @python_scope
    def fill(self, val):
        for v in self.members:
            v.fill(val)

    def initialize_host_accessors(self):
        for v in self.members:
            v.initialize_host_accessors()

    def get_member_field(self, key):
        return self.field_dict[key]

    @python_scope
    def from_numpy(self, array_dict):
        for k, v in self.items:
            v.from_numpy(array_dict[k])

    @python_scope
    def from_torch(self, array_dict):
        for k, v in self.items:
            v.from_torch(array_dict[k])

    @python_scope
    def to_numpy(self):
        return {k: v.to_numpy() for k, v in self.items}

    @python_scope
    def to_torch(self, device=None):
        return {k: v.to_torch(device=device) for k, v in self.items}

    @python_scope
    def __setitem__(self, indices,
                    element):  # TODO(changyu): handle reordering case
        self.initialize_host_accessors()
        self[indices].set_entries(element)

    @python_scope
    def __getitem__(self, indices):  # TODO(changyu): handle reordering case
        self.initialize_host_accessors()
        # scalar fields does not instantiate SNodeHostAccess by default
        entries = {
            k: v.host_access(self.pad_key(indices))[0] if isinstance(
                v, ScalarField) else v[indices]
            for k, v in self.items
        }
        return Struct(entries)


class MeshElement:
    def __init__(self, type):
        self.type = type
        self.layout = Layout.SOA
        self.attr_dict = {}

    def _SOA(self, soa=True):  # AOS/SOA
        self.layout = Layout.SOA if soa else Layout.AOS

    def _AOS(self, aos=True):
        self.layout = Layout.AOS if aos else Layout.SOA

    SOA = property(fget=_SOA)
    AOS = property(fget=_AOS)

    def place(
        self,
        members,
        reordering=MeshElementReorderingType.NonReordering,
        needs_grad=False,
    ):
        for key, dtype in members.items():
            self.attr_dict[key] = MeshAttrType(key, dtype, reordering,
                                               needs_grad)

    def build(self, size):
        field_dict = {}

        for key, attr in self.attr_dict.items():
            if isinstance(attr.dtype, CompoundType):
                field_dict[key] = attr.dtype.field(shape=None,
                                                   needs_grad=attr.needs_grad)
            else:
                field_dict[key] = impl.field(attr.dtype,
                                             shape=None,
                                             needs_grad=attr.needs_grad)

        if self.layout == Layout.SOA:
            for key, field in field_dict.items():
                impl.root.dense(impl.indices(0), size).place(field)
                if self.attr_dict[key].needs_grad:
                    impl.root.dense(impl.indices(0), size).place(field.grad)
        else:
            impl.root.dense(impl.indices(0),
                            size).place(*tuple(field_dict.values()))
            grads = []
            for key, field in field_dict.items():
                if self.attr_dict[key].needs_gard: grads.append(field.grad)
            impl.root.dense(impl.indices(0), size).place(*grads)

        return MeshElementField(self.type, self.attr_dict, field_dict)


# Define the instance of the Mesh Type, stores the field (type and data) info
class MeshInstance:
    def __init__(self, type):
        self.type = type
        self.mesh_ptr = _ti_core.create_mesh()

    def set_owned_offset(self, element_type: MeshElementType,
                         owned_offset: ScalarField):
        _ti_core.set_owned_offset(self.mesh_ptr, element_type,
                                  owned_offset.vars[0].ptr.snode())

    def set_total_offset(self, element_type: MeshElementType,
                         total_offset: ScalarField):
        _ti_core.set_total_offset(self.mesh_ptr, element_type,
                                  total_offset.vars[0].ptr.snode())

    def set_num_patches(self, num_patches: int):
        _ti_core.set_num_patches(self.mesh_ptr, num_patches)

    def add_mesh_attribute(self, element_type, snode, reordering_type):
        _ti_core.add_mesh_attribute(self.mesh_ptr, element_type, snode,
                                    reordering_type)


# Define the Mesh Type, stores the field type info
class MeshType:
    def __init__(self, topology):
        self.topology = topology
        self.verts = MeshElement(MeshElementType.Vertex)
        self.edges = MeshElement(MeshElementType.Edge)
        self.faces = MeshElement(MeshElementType.Face)
        if topology == MeshTopology.Tetrahedron:
            self.cells = MeshElement(MeshElementType.Cell)

    def build(self, size: tuple):
        instance = MeshInstance(self)
        if size[0] > 0:
            instance.verts = self.verts.build(size[0])
        if size[1] > 0:
            instance.edges = self.edges.build(size[1])
        if size[2] > 0:
            instance.faces = self.faces.build(size[2])
        if self.topology == MeshTopology.Tetrahedron and size[3] > 0:
            instance.cells = self.cells.build(size[3])
        return instance


# Mesh First Class
class Mesh:
    def __init__(self):
        pass

    @property
    def non_reordering():
        return MeshElementReorderingType.NonReordering

    @property
    def reordering():
        return MeshElementReorderingType.Reordering

    @property
    def surface_first():
        return MeshElementReorderingType.SurfaceFirst

    @property
    def cell_first():
        return MeshElementReorderingType.CellFirst

    @staticmethod
    def Tet():
        return MeshType(MeshTopology.Tetrahedron)

    @staticmethod
    def Tri():
        return MeshType(MeshTopology.Triangle)


def TriMesh():
    return Mesh.Tri()


def TetMesh():
    return Mesh.Tet()


class MeshRelationAccess:
    def __init__(self, mesh: MeshInstance, from_index: impl.Expr,
                 to_element_type: MeshElementType):
        self.mesh = mesh
        self.from_index = from_index
        self.to_element_type = to_element_type

        self.is_taichi_class = True

    @property
    def size(self):
        return impl.Expr(
            _ti_core.get_relation_size(self.mesh.mesh_ptr, self.from_index.ptr,
                                       self.to_element_type))
