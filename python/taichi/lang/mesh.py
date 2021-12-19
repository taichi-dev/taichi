import json

import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.field import Field, ScalarField
from taichi.lang.matrix import MatrixField, _IntermediateMatrix
from taichi.lang.struct import StructField
from taichi.lang.util import python_scope
from taichi.types import CompoundType

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
        return _IntermediateMatrix(self.n, self.m, self.host_access(key))


class MeshElementField:
    def __init__(self, mesh_instance, _type, attr_dict, field_dict, g2r_field):
        self.mesh = mesh_instance
        self._type = _type
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
                            self._type, self.g2r_field)
                    elif isinstance(self.field_dict[key], MatrixField):
                        self.getter_dict[key] = MeshReorderedMatrixFieldProxy(
                            self.field_dict[key], self.mesh.mesh_ptr,
                            self._type, self.g2r_field)
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

    SOA = property(fget=_SOA)
    AOS = property(fget=_AOS)

    def place(
        self,
        members,
        reorder=False,
        needs_grad=False,
    ):
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


class MeshMetadata:
    def __init__(self, filename):
        with open(filename, "r") as fi:
            data = json.loads(fi.read())

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
                dtype=ti.i32, shape=self.num_patches + 1)
            self.element_fields[element_type]["total"] = impl.field(
                dtype=ti.i32, shape=self.num_patches + 1)
            self.element_fields[element_type]["l2g"] = impl.field(
                dtype=ti.i32, shape=element["l2g_mapping"].shape[0])
            self.element_fields[element_type]["l2r"] = impl.field(
                dtype=ti.i32, shape=element["l2r_mapping"].shape[0])
            self.element_fields[element_type]["g2r"] = impl.field(
                dtype=ti.i32, shape=element["g2r_mapping"].shape[0])

        for relation in data["relations"]:
            from_order = relation["from_order"]
            to_order = relation["to_order"]
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            self.relation_fields[rel_type] = {}
            self.relation_fields[rel_type]["value"] = impl.field(
                dtype=ti.i32, shape=len(relation["value"]))
            if from_order <= to_order:
                self.relation_fields[rel_type]["offset"] = impl.field(
                    dtype=ti.i32, shape=len(relation["offset"]))

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
                self.relation_fields[rel_type]["offset"].from_numpy(
                    np.array(relation["offset"]))

        self.attrs = {}
        self.attrs["x"] = np.array(data["attrs"]["x"]).reshape(-1, 3)


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

    def build(self, metadata: MeshMetadata):
        instance = MeshInstance(self)
        instance.fields = {}

        instance.set_num_patches(metadata.num_patches)

        for element in self.elements:
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

        for relation in self.relations:
            from_order = element_order(relation[0])
            to_order = element_order(relation[1])
            rel_type = MeshRelationType(
                relation_by_orders(from_order, to_order))
            if from_order <= to_order:
                instance.set_relation_dynamic(
                    rel_type, metadata.relation_fields[rel_type]["value"],
                    metadata.relation_fields[rel_type]["offset"])
            else:
                instance.set_relation_fixed(
                    rel_type, metadata.relation_fields[rel_type]["value"])

        if "x" in instance.verts.attr_dict:  # pylint: disable=E1101
            instance.verts.x.from_numpy(metadata.attrs["x"])  # pylint: disable=E1101

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

    @staticmethod
    def load_meta(filename):
        return MeshMetadata(filename)


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
                    _IntermediateMatrix(attr.n, attr.m, [
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
            _ti_core.get_relation_size(self.mesh.mesh_ptr, self.from_index.ptr,
                                       self.to_element_type))

    def subscript(self, *indices):
        assert len(indices) == 1
        entry_expr = _ti_core.get_relation_access(self.mesh.mesh_ptr,
                                                  self.from_index.ptr,
                                                  self.to_element_type,
                                                  impl.Expr(indices[0]).ptr)
        entry_expr.type_check()
        return MeshElementFieldProxy(self.mesh, self.to_element_type,
                                     entry_expr)
