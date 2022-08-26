import ast
import json

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field, ScalarField
from taichi.lang.matrix import Matrix, MatrixField, _MatrixFieldElement
from taichi.lang.struct import StructField
from taichi.lang.util import python_scope
from taichi.types import u16, u32
from taichi.types.compound_types import CompoundType

from taichi import lang

MeshTopology = _ti_core.MeshTopology
MeshElementType = _ti_core.MeshElementType
MeshRelationType = _ti_core.MeshRelationType
ConvType = _ti_core.ConvType
element_order = _ti_core.element_order
from_end_element_order = _ti_core.from_end_element_order
to_end_element_order = _ti_core.to_end_element_order
relation_by_orders = _ti_core.relation_by_orders
inverse_relation = _ti_core.inverse_relation
element_type_name = _ti_core.element_type_name


class MeshAttrType:
    def __init__(self, name, dtype, reorder, needs_grad):
        self.name = name
        self.dtype = dtype
        self.reorder = reorder
        self.needs_grad = needs_grad


class MeshReorderedScalarFieldProxy(ScalarField):
    def __init__(self, field: ScalarField, mesh_ptr: _ti_core.MeshPtr,
                 element_type: MeshElementType, g2r_field: ScalarField):
        self.vars = field.vars
        self.host_accessors = field.host_accessors
        self.grad = field.grad

        self.mesh_ptr = mesh_ptr
        self.element_type = element_type
        self.g2r_field = g2r_field

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        key = self.g2r_field[key]
        self.host_accessors[0].setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        key = self.g2r_field[key]
        return self.host_accessors[0].getter(*self._pad_key(key))


class MeshReorderedMatrixFieldProxy(MatrixField):
    def __init__(self, field: MatrixField, mesh_ptr: _ti_core.MeshPtr,
                 element_type: MeshElementType, g2r_field: ScalarField):
        self.vars = field.vars
        self.host_accessors = field.host_accessors
        self.grad = field.grad
        self.n = field.n
        self.m = field.m
        self.dynamic_index_stride = field.dynamic_index_stride

        self.mesh_ptr = mesh_ptr
        self.element_type = element_type
        self.g2r_field = g2r_field

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self[key]._set_entries(value)

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        key = self.g2r_field[key]
        key = self._pad_key(key)
        return Matrix(self._host_access(key), is_ref=True)


class MeshElementField:
    def __init__(self, mesh_instance, _type, attr_dict, field_dict, g2r_field):
        self.mesh = mesh_instance
        self._type = _type
        self.attr_dict = attr_dict
        self.field_dict = field_dict
        self.g2r_field = g2r_field

        self._register_fields()

    @property
    def keys(self):
        return list(self.field_dict.keys())

    @property
    def _members(self):
        return list(self.field_dict.values())

    @property
    def _items(self):
        return self.field_dict.items()

    @staticmethod
    def _make_getter(key):
        def getter(self):
            if key not in self.getter_dict:
                if self.attr_dict[key].reorder:
                    if isinstance(self.field_dict[key], ScalarField):
                        self.getter_dict[key] = MeshReorderedScalarFieldProxy(
                            self.field_dict[key], self.mesh.mesh_ptr,
                            self._type, self.g2r_field)
                    elif isinstance(self.field_dict[key], MatrixField):
                        self.getter_dict[key] = MeshReorderedMatrixFieldProxy(
                            self.field_dict[key], self.mesh.mesh_ptr,
                            self._type, self.g2r_field)
                else:
                    self.getter_dict[key] = self.field_dict[key]
            """Get an entry from custom struct by name."""
            return self.getter_dict[key]

        return getter

    def _register_fields(self):
        self.getter_dict = {}
        for k in self.keys:
            setattr(MeshElementField, k,
                    property(fget=MeshElementField._make_getter(k)))

    def _get_field_members(self):
        field_members = []
        for m in self._members:
            assert isinstance(m, Field)
            field_members += m._get_field_members()
        return field_members

    @python_scope
    def copy_from(self, other):
        assert isinstance(other, Field)
        assert set(self.keys) == set(other.keys)
        for k in self.keys:
            self.field_dict[k].copy_from(other[k])

    @python_scope
    def fill(self, val):
        for v in self._members:
            v.fill(val)

    def _initialize_host_accessors(self):
        for v in self._members:
            v._initialize_host_accessors()

    def get_member_field(self, key):
        return self.field_dict[key]

    @python_scope
    def from_numpy(self, array_dict):
        for k, v in self._items:
            v.from_numpy(array_dict[k])

    @python_scope
    def from_torch(self, array_dict):
        for k, v in self._items:
            v.from_torch(array_dict[k])

    @python_scope
    def from_paddle(self, array_dict):
        for k, v in self._items:
            v.from_paddle(array_dict[k])

    @python_scope
    def to_numpy(self):
        return {k: v.to_numpy() for k, v in self._items}

    @python_scope
    def to_torch(self, device=None):
        return {k: v.to_torch(device=device) for k, v in self._items}

    @python_scope
    def to_paddle(self, place=None):
        return {k: v.to_paddle(place=place) for k, v in self._items}

    @python_scope
    def __len__(self):
        return _ti_core.get_num_elements(self.mesh.mesh_ptr, self._type)


class MeshElement:
    def __init__(self, _type, builder):
        self.builder = builder
        self._type = _type
        self.layout = Layout.SOA
        self.attr_dict = {}

    def _SOA(self, soa=True):  # AOS/SOA
        self.layout = Layout.SOA if soa else Layout.AOS

    def _AOS(self, aos=True):
        self.layout = Layout.AOS if aos else Layout.SOA

    SOA = property(fset=_SOA)
    """ Set `True` for SOA (structure of arrays) layout.
    """
    AOS = property(fset=_AOS)
    """ Set `True` for AOS (array of structures) layout.
    """

    def place(
        self,
        members,
        reorder=False,
        needs_grad=False,
    ):
        """Declares mesh attributes for the mesh element in current mesh builder.

        Args:
        members (Dict[str, Union[PrimitiveType, MatrixType]]): \
            names and types for element attributes.
        reorder: True if reorders the internal memory for coalesced data access within mesh-for loop.
        needs_grad: True if needs to record grad.

        Example::
        >>> vec3 = ti.types.vector(3, ti.f32)
        >>> mesh = ti.TriMesh()
        >>> mesh.faces.place({'area' : ti.f32}) # declares a mesh attribute `area` for each face element.
        >>> mesh.verts.place({'pos' : vec3}, reorder=True) # declares a mesh attribute `pos` for each vertex element, and reorder it in memory.
        """
        self.builder.elements.add(self._type)
        for key, dtype in members.items():
            if key in {'verts', 'edges', 'faces', 'cells'}:
                raise TaichiSyntaxError(
                    f"'{key}' cannot use as attribute name. It has been reserved as ti.Mesh's keyword."
                )
            self.attr_dict[key] = MeshAttrType(key, dtype, reorder, needs_grad)

    def build(self, mesh_instance, size, g2r_field):
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
                impl.root.dense(impl.axes(0), size).place(field)
                if self.attr_dict[key].needs_grad:
                    impl.root.dense(impl.axes(0), size).place(field.grad)
        elif len(field_dict) > 0:
            impl.root.dense(impl.axes(0),
                            size).place(*tuple(field_dict.values()))
            grads = []
            for key, field in field_dict.items():
                if self.attr_dict[key].needs_grad:
                    grads.append(field.grad)
            if len(grads) > 0:
                impl.root.dense(impl.axes(0), size).place(*grads)

        return MeshElementField(mesh_instance, self._type, self.attr_dict,
                                field_dict, g2r_field)

    def link(self, element):
        """Explicitly declares the element-element connectivity for compiler to pre-generate relation data.

        Args:
            element (MeshElement): mesh element in the same builder to represent the to-end of connectivity.

        Example::
            >>> mesh = ti.TriMesh()
            >>> mesh.faces.link(mesh.verts) # declares F-V connectivity
            >>> mesh.verts.link(mesh.verts) # declares V-V connectivity
        """
        assert isinstance(element, MeshElement)
        assert element.builder == self.builder
        self.builder.relations.add(tuple([self._type, element._type]))
        self.builder.elements.add(self._type)
        self.builder.elements.add(element._type)


# Define the instance of the Mesh Type, stores the field (type and data) info
class MeshInstance:
    def __init__(self, _type):
        self._type = _type
        self.mesh_ptr = _ti_core.create_mesh()
        self.relation_set = set()

    def set_owned_offset(self, element_type: MeshElementType,
                         owned_offset: ScalarField):
        _ti_core.set_owned_offset(self.mesh_ptr, element_type,
                                  owned_offset.vars[0].ptr.snode())

    def set_total_offset(self, element_type: MeshElementType,
                         total_offset: ScalarField):
        _ti_core.set_total_offset(self.mesh_ptr, element_type,
                                  total_offset.vars[0].ptr.snode())

    def set_index_mapping(self, element_type: MeshElementType,
                          conv_type: ConvType, mapping: ScalarField):
        _ti_core.set_index_mapping(self.mesh_ptr, element_type, conv_type,
                                   mapping.vars[0].ptr.snode())

    def set_num_patches(self, num_patches: int):
        _ti_core.set_num_patches(self.mesh_ptr, num_patches)

    def set_patch_max_element_num(self, element_type: MeshElementType,
                                  max_element_num: int):
        _ti_core.set_patch_max_element_num(self.mesh_ptr, element_type,
                                           max_element_num)

    def set_relation_fixed(self, rel_type: MeshRelationType,
                           value: ScalarField):
        self.relation_set.add(rel_type)
        _ti_core.set_relation_fixed(self.mesh_ptr, rel_type,
                                    value.vars[0].ptr.snode())

    def set_relation_dynamic(self, rel_type: MeshRelationType,
                             value: ScalarField, patch_offset: ScalarField,
                             offset: ScalarField):
        self.relation_set.add(rel_type)
        _ti_core.set_relation_dynamic(self.mesh_ptr, rel_type,
                                      value.vars[0].ptr.snode(),
                                      patch_offset.vars[0].ptr.snode(),
                                      offset.vars[0].ptr.snode())

    def add_mesh_attribute(self, element_type, snode, reorder_type):
        _ti_core.add_mesh_attribute(self.mesh_ptr, element_type, snode,
                                    reorder_type)

    def get_relation_size(self, from_index, to_element_type):
        return _ti_core.get_relation_size(self.mesh_ptr, from_index.ptr,
                                          to_element_type)

    def get_relation_access(self, from_index, to_element_type,
                            neighbor_idx_ptr):
        return _ti_core.get_relation_access(self.mesh_ptr, from_index.ptr,
                                            to_element_type, neighbor_idx_ptr)

    def update_relation(self, from_order, to_order):
        rel_type = MeshRelationType(relation_by_orders(from_order, to_order))
        if rel_type not in self.relation_set:
            meta = self.patcher.get_relation_meta(from_order, to_order)

            def fun(arr, dtype):
                field = impl.field(dtype=dtype, shape=arr.shape)
                field.from_numpy(arr)
                return field

            if from_order <= to_order:
                self.set_relation_dynamic(rel_type, fun(meta["value"], u16),
                                          fun(meta["patch_offset"], u32),
                                          fun(meta["offset"], u16))
            else:
                self.set_relation_fixed(rel_type, fun(meta["value"], u16))


class MeshMetadata:
    def __init__(self, data):
        self.num_patches = data["num_patches"]

        self.element_fields = {}
        self.relation_fields = {}
        self.num_elements = {}
        self.max_num_per_patch = {}

        for element in data["elements"]:
            element_type = MeshElementType(element["order"])
            self.num_elements[element_type] = element["num"]
            self.max_num_per_patch[element_type] = element["max_num_per_patch"]

            element["l2g_mapping"] = np.array(element["l2g_mapping"])
            element["l2r_mapping"] = np.array(element["l2r_mapping"])
            element["g2r_mapping"] = np.array(element["g2r_mapping"])
            self.element_fields[element_type] = {}
            self.element_fields[element_type]["owned"] = impl.field(
                dtype=u32, shape=self.num_patches + 1)
            self.element_fields[element_type]["total"] = impl.field(
                dtype=u32, shape=self.num_patches + 1)
            self.element_fields[element_type]["l2g"] = impl.field(
                dtype=u32, shape=element["l2g_mapping"].shape[0])
            self.element_fields[element_type]["l2r"] = impl.field(
                dtype=u32, shape=element["l2r_mapping"].shape[0])
            self.element_fields[element_type]["g2r"] = impl.field(
                dtype=u32, shape=element["g2r_mapping"].shape[0])

        for relation in data["relations"]:
            from_order = relation["from_order"]
            to_order = relation["to_order"]
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            self.relation_fields[rel_type] = {}
            self.relation_fields[rel_type]["value"] = impl.field(
                dtype=u16, shape=len(relation["value"]))
            if from_order <= to_order:
                self.relation_fields[rel_type]["offset"] = impl.field(
                    dtype=u16, shape=len(relation["offset"]))
                self.relation_fields[rel_type]["patch_offset"] = impl.field(
                    dtype=u32, shape=len(relation["patch_offset"]))
            self.relation_fields[rel_type]["from_order"] = from_order
            self.relation_fields[rel_type]["to_order"] = to_order

        for element in data["elements"]:
            element_type = MeshElementType(element["order"])
            self.element_fields[element_type]["owned"].from_numpy(
                np.array(element["owned_offsets"]))
            self.element_fields[element_type]["total"].from_numpy(
                np.array(element["total_offsets"]))
            self.element_fields[element_type]["l2g"].from_numpy(
                element["l2g_mapping"])
            self.element_fields[element_type]["l2r"].from_numpy(
                element["l2r_mapping"])
            self.element_fields[element_type]["g2r"].from_numpy(
                element["g2r_mapping"])

        for relation in data["relations"]:
            from_order = relation["from_order"]
            to_order = relation["to_order"]
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            self.relation_fields[rel_type]["value"].from_numpy(
                np.array(relation["value"]))
            if from_order <= to_order:
                self.relation_fields[rel_type]["patch_offset"].from_numpy(
                    np.array(relation["patch_offset"]))
                self.relation_fields[rel_type]["offset"].from_numpy(
                    np.array(relation["offset"]))

        self.attrs = {}
        self.attrs["x"] = np.array(data["attrs"]["x"]).reshape(-1, 3)
        if "patcher" in data:
            self.patcher = data["patcher"]
        else:
            self.patcher = None


# Define the Mesh Type, stores the field type info
class MeshBuilder:
    def __init__(self, topology):
        if not lang.misc.is_extension_supported(impl.current_cfg().arch,
                                                lang.extension.mesh):
            raise Exception('Backend ' + str(impl.current_cfg().arch) +
                            ' doesn\'t support MeshTaichi extension')

        self.topology = topology
        self.verts = MeshElement(MeshElementType.Vertex, self)
        self.edges = MeshElement(MeshElementType.Edge, self)
        self.faces = MeshElement(MeshElementType.Face, self)
        if topology == MeshTopology.Tetrahedron:
            self.cells = MeshElement(MeshElementType.Cell, self)

        self.elements = set()
        self.relations = set()

        impl.current_cfg().use_mesh = True

    def build(self, metadata: MeshMetadata):
        """Build and instantiate mesh from model meta data

        Use the following external lib to generate meta data:
        https://github.com/BillXu2000/meshtaichi_patcher

        Args:
            metadata : model meta data.

        Returns:
            The mesh instance class.
        """
        instance = MeshInstance(self)
        instance.fields = {}

        instance.set_num_patches(metadata.num_patches)

        for element in metadata.element_fields:
            self.elements.add(element)
            _ti_core.set_num_elements(instance.mesh_ptr, element,
                                      metadata.num_elements[element])
            instance.set_patch_max_element_num(
                element, metadata.max_num_per_patch[element])

            element_name = element_type_name(element)
            setattr(
                instance, element_name,
                getattr(self, element_name).build(
                    instance, metadata.num_elements[element],
                    metadata.element_fields[element]["g2r"]))
            instance.fields[element] = getattr(instance, element_name)

            instance.set_owned_offset(
                element, metadata.element_fields[element]["owned"])
            instance.set_total_offset(
                element, metadata.element_fields[element]["total"])
            instance.set_index_mapping(element, ConvType.l2g,
                                       metadata.element_fields[element]["l2g"])
            instance.set_index_mapping(element, ConvType.l2r,
                                       metadata.element_fields[element]["l2r"])
            instance.set_index_mapping(element, ConvType.g2r,
                                       metadata.element_fields[element]["g2r"])

        for rel_type in metadata.relation_fields:
            from_order = metadata.relation_fields[rel_type]["from_order"]
            to_order = metadata.relation_fields[rel_type]["to_order"]
            if from_order <= to_order:
                instance.set_relation_dynamic(
                    rel_type, metadata.relation_fields[rel_type]["value"],
                    metadata.relation_fields[rel_type]["patch_offset"],
                    metadata.relation_fields[rel_type]["offset"])
            else:
                instance.set_relation_fixed(
                    rel_type, metadata.relation_fields[rel_type]["value"])

        if "x" in instance.verts.attr_dict:  # pylint: disable=E1101
            instance.verts.x.from_numpy(metadata.attrs["x"])  # pylint: disable=E1101

        instance.patcher = metadata.patcher

        return instance


# Mesh First Class
class Mesh:
    """The Mesh type class.

    ti.Mesh offers first-class support for triangular/tetrahedral meshes
    and allows efficient computation on these irregular data structures,
    only available for backends supporting `ti.extension.mesh`.

    Related to https://github.com/taichi-dev/taichi/issues/3608
    """
    def __init__(self):
        pass

    @staticmethod
    def Tet():
        """Create a tetrahedron mesh (a set of vert/edge/face/cell elements, attributes, and connectivity) builder.

        Returns:
            An instance of mesh builder.
        """
        return MeshBuilder(MeshTopology.Tetrahedron)

    @staticmethod
    def Tri():
        """Create a triangle mesh (a set of vert/edge/face elements, attributes, and connectivity) builder.

        Returns:
            An instance of mesh builder.
        """
        return MeshBuilder(MeshTopology.Triangle)

    @staticmethod
    def load_meta(filename):
        with open(filename, "r") as fi:
            data = json.loads(fi.read())
        return MeshMetadata(data)

    @staticmethod
    def generate_meta(data):
        return MeshMetadata(data)

    class RelationVisitor(ast.NodeVisitor):
        # TODO: only works for simple cases

        def __init__(self, ctx):
            self.vars = {}
            self.visits = []
            self.ctx = ctx

        def visit_For(self, node):
            if isinstance(node.iter, ast.Attribute):
                value = node.iter.value
                if isinstance(value, ast.Name):
                    if value.id in self.ctx.global_vars:
                        var = self.ctx.global_vars[value.id]
                        if isinstance(var, MeshInstance):
                            self.vars[node.target.id] = [var, node.iter.attr]
            if isinstance(node.iter, ast.Name):
                if node.iter.id in self.ctx.global_vars:
                    var = self.ctx.global_vars[node.iter.id]
                    if isinstance(var, MeshElementField):
                        self.vars[node.target.id] = [
                            var.mesh, element_type_name(var._type)
                        ]
            ast.NodeVisitor.generic_visit(self, node)

        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.vars:
                        self.vars[node.targets[0].id] = self.vars[
                            node.value.id]
            ast.NodeVisitor.generic_visit(self, node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name):
                if node.value.id in self.vars:
                    self.visits.append(self.vars[node.value.id] + [node.attr])
            ast.NodeVisitor.generic_visit(self, node)

    @staticmethod
    def update_relation(tree, ctx):
        x = Mesh.RelationVisitor(ctx)
        x.visit(tree)
        name_to_order = {"verts": 0, "edges": 1, "faces": 2, "cells": 3}
        for visit in x.visits:
            if visit[1] in name_to_order and visit[2] in name_to_order:
                visit[0].update_relation(name_to_order[visit[1]],
                                         name_to_order[visit[2]])


def TriMesh():
    """Create a triangle mesh (a set of vert/edge/face elements, attributes, and connectivity) builder.

    Returns:
        An instance of mesh builder.
    """
    return Mesh.Tri()


def TetMesh():
    """Create a tetrahedron mesh (a set of vert/edge/face/cell elements, attributes, and connectivity) builder.

    Returns:
        An instance of mesh builder.
    """
    return Mesh.Tet()


class MeshElementFieldProxy:
    def __init__(self, mesh: MeshInstance, element_type: MeshElementType,
                 entry_expr: impl.Expr):
        self.mesh = mesh
        self.element_type = element_type
        self.entry_expr = entry_expr

        element_field = self.mesh.fields[self.element_type]
        for key, attr in element_field.field_dict.items():
            global_entry_expr = impl.Expr(
                _ti_core.get_index_conversion(
                    self.mesh.mesh_ptr, element_type, entry_expr,
                    ConvType.l2r if element_field.attr_dict[key].reorder else
                    ConvType.l2g))  # transform index space
            global_entry_expr_group = impl.make_expr_group(
                *tuple([global_entry_expr]))
            if isinstance(attr, MatrixField):
                setattr(self, key,
                        _MatrixFieldElement(attr, global_entry_expr_group))
            elif isinstance(attr, StructField):
                raise RuntimeError('ti.Mesh has not support StructField yet')
            else:  # isinstance(attr, Field)
                var = attr._get_field_members()[0].ptr
                setattr(
                    self, key,
                    impl.Expr(
                        _ti_core.subscript(
                            var, global_entry_expr_group,
                            impl.get_runtime().get_current_src_info())))

        for element_type in self.mesh._type.elements:
            setattr(self, element_type_name(element_type),
                    impl.mesh_relation_access(self.mesh, self, element_type))

    @property
    def ptr(self):
        return self.entry_expr

    @property
    def id(self):  # return the global non-reordered index
        l2g_expr = impl.Expr(
            _ti_core.get_index_conversion(self.mesh.mesh_ptr,
                                          self.element_type, self.entry_expr,
                                          ConvType.l2g))
        return l2g_expr


class MeshRelationAccessProxy:
    def __init__(self, mesh: MeshInstance, from_index: impl.Expr,
                 to_element_type: MeshElementType):
        self.mesh = mesh
        self.from_index = from_index
        self.to_element_type = to_element_type

    @property
    def size(self):
        return impl.Expr(
            self.mesh.get_relation_size(self.from_index, self.to_element_type))

    def subscript(self, *indices):
        assert len(indices) == 1
        entry_expr = self.mesh.get_relation_access(self.from_index,
                                                   self.to_element_type,
                                                   impl.Expr(indices[0]).ptr)
        entry_expr.type_check(impl.get_runtime().prog.config)
        return MeshElementFieldProxy(self.mesh, self.to_element_type,
                                     entry_expr)


__all__ = ["Mesh", "TetMesh", "TriMesh"]
