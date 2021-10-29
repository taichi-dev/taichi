import json

import numpy as np
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.field import Field, ScalarField
from taichi.lang.matrix import Matrix, MatrixField
from taichi.lang.struct import Struct, StructField
from taichi.lang.types import CompoundType
from taichi.lang.util import (cook_dtype, has_pytorch, is_taichi_class,
                              python_scope, taichi_scope, to_pytorch_type)

import taichi as ti

MeshTopology = _ti_core.MeshTopology
MeshElementType = _ti_core.MeshElementType
MeshRelationType = _ti_core.MeshRelationType
ConvType = _ti_core.ConvType
element_order = _ti_core.element_order
from_end_element_order = _ti_core.from_end_element_order
to_end_element_order = _ti_core.to_end_element_order
relation_by_orders = _ti_core.relation_by_orders
inverse_relation = _ti_core.inverse_relation


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
        self.initialize_host_accessors()
        key = self.g2r_field[key]
        self.host_accessors[0].setter(value, *self.pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self.initialize_host_accessors()
        key = self.g2r_field[key]
        return self.host_accessors[0].getter(*self.pad_key(key))


class MeshReorderedMatrixFieldProxy(MatrixField):
    def __init__(self, field: MatrixField, mesh_ptr: _ti_core.MeshPtr,
                 element_type: MeshElementType, g2r_field: ScalarField):
        self.vars = field.vars
        self.host_accessors = field.host_accessors
        self.grad = field.grad
        self.n = field.n
        self.m = field.m

        self.mesh_ptr = mesh_ptr
        self.element_type = element_type
        self.g2r_field = g2r_field

    @python_scope
    def __setitem__(self, key, value):
        self.initialize_host_accessors()
        self[key].set_entries(value)

    @python_scope
    def __getitem__(self, key):
        self.initialize_host_accessors()
        key = self.g2r_field[key]
        key = self.pad_key(key)
        return Matrix.with_entries(self.n, self.m, self.host_access(key))


class MeshElementField:
    def __init__(self, mesh_instance, type, attr_dict, field_dict, g2r_field):
        self.mesh = mesh_instance
        self.type = type
        self.attr_dict = attr_dict
        self.field_dict = field_dict
        self.g2r_field = g2r_field

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
            if key not in self.getter_dict:
                if self.attr_dict[key].reorder:
                    if isinstance(self.field_dict[key], ScalarField):
                        self.getter_dict[key] = MeshReorderedScalarFieldProxy(
                            self.field_dict[key], self.mesh.mesh_ptr,
                            self.type, self.g2r_field)
                    elif isinstance(self.field_dict[key], MatrixField):
                        self.getter_dict[key] = MeshReorderedMatrixFieldProxy(
                            self.field_dict[key], self.mesh.mesh_ptr,
                            self.type, self.g2r_field)
                else:
                    self.getter_dict[key] = self.field_dict[key]
            """Get an entry from custom struct by name."""
            _taichi_skip_traceback = 1
            return self.getter_dict[key]

        return getter

    def register_fields(self):
        self.getter_dict = {}
        for k in self.keys:
            setattr(MeshElementField, k,
                    property(fget=MeshElementField.make_getter(k)))

    def get_field_members(self):
        field_members = []
        for m in self.members:
            assert isinstance(m, Field)
            field_members += m.get_field_members()
        return field_members

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
    def __len__(self):
        return _ti_core.get_num_elements(self.mesh.mesh_ptr, self.type)


class MeshElement:
    def __init__(self, type, builder):
        self.builder = builder
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
        reorder=False,
        needs_grad=False,
    ):
        self.builder.elements.add(self.type)
        for key, dtype in members.items():
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
                if self.attr_dict[key].needs_grad: grads.append(field.grad)
            if len(grads) > 0:
                impl.root.dense(impl.axes(0), size).place(*grads)

        return MeshElementField(mesh_instance, self.type, self.attr_dict,
                                field_dict, g2r_field)

    def link(self, element):
        assert isinstance(element, MeshElement)
        assert element.builder == self.builder
        self.builder.relations.add(tuple([self.type, element.type]))
        self.builder.elements.add(self.type)
        self.builder.elements.add(element.type)


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
        _ti_core.set_relation_fixed(self.mesh_ptr, rel_type,
                                    value.vars[0].ptr.snode())

    def set_relation_dynamic(self, rel_type: MeshRelationType,
                             value: ScalarField, offset: ScalarField):
        _ti_core.set_relation_dynamic(self.mesh_ptr, rel_type,
                                      value.vars[0].ptr.snode(),
                                      offset.vars[0].ptr.snode())

    def add_mesh_attribute(self, element_type, snode, reorder_type):
        _ti_core.add_mesh_attribute(self.mesh_ptr, element_type, snode,
                                    reorder_type)


# Define the Mesh Type, stores the field type info
class MeshBuilder:
    def __init__(self, topology):
        if not ti.is_extension_supported(ti.cfg.arch, ti.extension.mesh):
            raise Exception('Backend ' + str(ti.cfg.arch) +
                            ' doesn\'t support MeshTaichi extension')

        self.topology = topology
        self.verts = MeshElement(MeshElementType.Vertex, self)
        self.edges = MeshElement(MeshElementType.Edge, self)
        self.faces = MeshElement(MeshElementType.Face, self)
        if topology == MeshTopology.Tetrahedron:
            self.cells = MeshElement(MeshElementType.Cell, self)

        self.elements = set()
        self.relations = set()

    def build(self, filename):
        instance = MeshInstance(self)
        instance.fields = {}

        inv_map = {
            MeshElementType.Vertex: "verts",
            MeshElementType.Edge: "edges",
            MeshElementType.Face: "faces",
            MeshElementType.Cell: "cells"
        }

        with open(filename, "r") as fi:
            data = json.loads(fi.read())

        num_patches = data["num_patches"]
        instance.set_num_patches(num_patches)
        element_fields = {}
        relation_fields = {}
        for element in data["elements"]:
            element_type = MeshElementType(element["order"])
            num_elements = element["num"]
            _ti_core.set_num_elements(instance.mesh_ptr, element_type,
                                      num_elements)
            instance.set_patch_max_element_num(element_type,
                                               element["max_num_per_patch"])

            element["l2g_mapping"] = np.array(element["l2g_mapping"])
            element["l2r_mapping"] = np.array(element["l2r_mapping"])
            element["g2r_mapping"] = np.array(element["g2r_mapping"])
            element_fields[element_type] = {}
            element_fields[element_type]["owned"] = impl.field(
                dtype=ti.u32, shape=num_patches + 1)
            element_fields[element_type]["total"] = impl.field(
                dtype=ti.u32, shape=num_patches + 1)
            element_fields[element_type]["l2g"] = impl.field(
                dtype=ti.u32, shape=element["l2g_mapping"].shape[0])
            element_fields[element_type]["l2r"] = impl.field(
                dtype=ti.u32, shape=element["l2r_mapping"].shape[0])
            element_fields[element_type]["g2r"] = impl.field(
                dtype=ti.i32, shape=element["g2r_mapping"].shape[0])

            element_name = inv_map[element_type]
            setattr(
                instance, element_name,
                getattr(self, element_name).build(
                    instance, num_elements,
                    element_fields[element_type]["g2r"]))
            instance.fields[element_type] = getattr(instance, element_name)

            instance.set_owned_offset(element_type,
                                      element_fields[element_type]["owned"])
            instance.set_total_offset(element_type,
                                      element_fields[element_type]["total"])
            instance.set_index_mapping(element_type, ConvType.l2g,
                                       element_fields[element_type]["l2g"])
            instance.set_index_mapping(element_type, ConvType.l2r,
                                       element_fields[element_type]["l2r"])
            instance.set_index_mapping(element_type, ConvType.g2r,
                                       element_fields[element_type]["g2r"])

        for relation in data["relations"]:
            from_order = relation["from_order"]
            to_order = relation["to_order"]
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            relation_fields[rel_type] = {}
            if from_order <= to_order:
                relation_fields[rel_type]["offset"] = impl.field(
                    dtype=ti.u32, shape=len(relation["offset"]))
                relation_fields[rel_type]["value"] = impl.field(
                    dtype=ti.u32, shape=len(relation["value"]))
                instance.set_relation_dynamic(
                    rel_type, relation_fields[rel_type]["value"],
                    relation_fields[rel_type]["offset"])
            else:
                relation_fields[rel_type]["value"] = impl.field(
                    dtype=ti.u32, shape=len(relation["value"]))
                instance.set_relation_fixed(rel_type,
                                            relation_fields[rel_type]["value"])
                relation_fields[rel_type]["value"].from_numpy(
                    np.array(relation["value"]))

        for element in data["elements"]:
            element_type = MeshElementType(element["order"])
            element_fields[element_type]["owned"].from_numpy(
                np.array(element["owned_offsets"]))
            element_fields[element_type]["total"].from_numpy(
                np.array(element["total_offsets"]))
            element_fields[element_type]["l2g"].from_numpy(
                element["l2g_mapping"])
            element_fields[element_type]["l2r"].from_numpy(
                element["l2r_mapping"])
            element_fields[element_type]["g2r"].from_numpy(
                element["g2r_mapping"])

        for relation in data["relations"]:
            from_order = relation["from_order"]
            to_order = relation["to_order"]
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            relation_fields[rel_type]["value"].from_numpy(
                np.array(relation["value"]))
            if from_order <= to_order:
                relation_fields[rel_type]["offset"].from_numpy(
                    np.array(relation["offset"]))

        if "x" in instance.verts.attr_dict:
            x = np.array(data["attrs"]["x"]).reshape(-1, 3)
            instance.verts.x.from_numpy(x)

        return instance


# Mesh First Class
class Mesh:
    def __init__(self):
        pass

    @staticmethod
    def Tet():
        return MeshBuilder(MeshTopology.Tetrahedron)

    @staticmethod
    def Tri():
        return MeshBuilder(MeshTopology.Triangle)


def TriMesh():
    return Mesh.Tri()


def TetMesh():
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
                setattr(
                    self, key,
                    Matrix.with_entries(attr.n, attr.m, [
                        impl.Expr(
                            _ti_core.subscript(e.ptr, global_entry_expr_group))
                        for e in attr.get_field_members()
                    ]))
            elif isinstance(attr, StructField):
                raise RuntimeError('ti.Mesh has not support StructField yet')
            else:  # isinstance(attr, Field)
                var = attr.get_field_members()[0].ptr
                setattr(
                    self, key,
                    impl.Expr(_ti_core.subscript(var,
                                                 global_entry_expr_group)))

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
            _ti_core.get_relation_size(self.mesh.mesh_ptr, self.from_index.ptr,
                                       self.to_element_type))

    def subscript(self, *indices):
        assert (len(indices) == 1)
        entry_expr = _ti_core.get_relation_access(self.mesh.mesh_ptr,
                                                  self.from_index.ptr,
                                                  self.to_element_type,
                                                  impl.Expr(indices[0]).ptr)
        return MeshElementFieldProxy(self.mesh, self.to_element_type,
                                     entry_expr)
