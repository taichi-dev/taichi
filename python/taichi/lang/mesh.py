from taichi.lang.kernel_impl import kernel
from typing import OrderedDict
import numpy as np
from enum import Enum
from taichi.core.util import ti_core as _ti_core
from taichi.lang.exception import InvalidOperationError, TaichiSyntaxError
from taichi.lang.field import Field, ScalarField
from taichi.lang.matrix import Matrix, MatrixField
from taichi.lang.kernel_arguments import template
from taichi.lang.snode import SNode
from taichi.lang.impl import *
from taichi.lang import impl
import taichi.core.primitive_types as primitive_type
from taichi.misc.mesh_loader import *
from taichi.lang.enums import Layout

MeshElementType = _ti_core.MeshElementType
MeshRelationType = _ti_core.MeshRelationType
MeshElementReorderingType = _ti_core.MeshElementReorderingType

def element_order(element_type):
    return int(element_type)

def from_end_element_order(rel):
    return int(rel) // 4

def to_end_element_order(rel):
    return int(rel) % 4

def relation_by_orders(from_order, to_order):
    return from_order * 4 + to_order

def inverse_relation(rel):
    return relation_by_orders(to_end_element_order(rel), from_end_element_order(rel))

class MeshAttribute:
    def __init__(self, name, field, reordering):
        self.name = name
        self.field = field
        self.reordering = reordering

class MeshElement:
    def __init__(self, type):
        self.type = type
        self.data_layout = Layout.SOA
        self.attributes = {}

    def place(self, name, field, reordering=MeshElementReorderingType.NonReordering):
        if isinstance(field, Field):
            for var in field.get_field_members():
                if var.ptr.snode() is not None:
                    raise RuntimeError(f'Field has been placed')
            self.attributes[name] = MeshAttribute(name, field, reordering)
        else:
            raise ValueError(f'{field} cannot be placed')

    def _SOA(self, soa=True): # AOS/SOA
        self.data_layout = Layout.SOA if soa else Layout.AOS
    def _AOS(self, aos=True):
        self.data_layout = Layout.AOS if aos else Layout.SOA
    
    SOA = property(fget = _SOA)
    AOS = property(fget = _AOS)

    def instance(self, mesh_inst, element_type):
        inst_attributes = []
        for name, attr in self.attributes.items():
            new_field = None
            if isinstance(attr.field, ScalarField):
                new_field = field(attr.field.dtype0)
            elif isinstance(attr.field, MatrixField):
                new_field = Matrix.field(attr.field.n, attr.field.m, attr.field.dtype0)
            mesh_inst.field_element_type[new_field] = element_type
            inst_attributes.append(MeshAttribute(name, new_field, attr.reordering))
            setattr(mesh_inst, name, new_field)
        return inst_attributes


# Global Relation used for patching, for a tri-mesh, we need FV, FE, FF, for a tet-mesh, we need CV, CE, CF, CC
# Also CF, FE, EV is initialized with input data
class MeshGlobalRelation:
    def __init__(self, relation):
        self._relation = relation
        self.validated = False
        self._order = [from_end_element_order(relation), to_end_element_order(relation)]
        self._fixed_relation = (self._order[0] > self._order[1])
        if self._fixed_relation:
            self.validate = self.validate_fixed
            self.from_numpy = self.from_numpy_fixed
        else:
            self.validate = self.validate_dynamic
            self.from_numpy = self.from_numpy_dynamic
    
    def validate_fixed(self, element_num):
        assert self._fixed_relation, "This Relation is not fixed type."
        self.value = field(primitive_type.u32)
        root.dense(indices(0, 1), 
        (element_num, 6 if self._relation is MeshRelationType.CE else self._order[0]+1)
        ).place(self.value)

        self.validated = True
        return self

    def validate_dynamic(self, num_offset, num_value):
        assert not self._fixed_relation, "This Relation is fixed type."
        self.offset = field(primitive_type.u32)
        self.value = field(primitive_type.u32)
        root.dense(indices(0), num_offset).place(self.offset)
        root.dense(indices(0), num_value).place(self.value)

        self.validated = True
        return self
    
    def validate_from_taichi_field(self, offset, value):
        assert not self._fixed_relation, "This Relation is fixed type."
        self.offset = offset
        self.value = value
        self.validated = True
        return self
    
    def from_numpy_fixed(self, np_value):
        assert self._fixed_relation, "This Relation is not fixed type."
        print(self.value.shape)
        self.value.from_numpy(np_value)
    
    def from_numpy_dynamic(self, np_offset, np_value):
        assert not self._fixed_relation, "This Relation is fixed type."
        self.offset.from_numpy(np_offset)
        self.value.from_numpy(np_value)

# Store a group of relations
class MeshGlobalRelationGroup:
    def __init__(self, highest_order):
        self._relations = {}
        highest_order = element_order(highest_order)
        for from_order in range(highest_order + 1):
            for to_order in range(highest_order + 1):
                rel_type = relation_by_orders(from_order, to_order)
                self._relations[rel_type] = MeshGlobalRelation(rel_type)
    
    def relation(self, type):
        return self._relations[type]

# Define the instance of the Mesh Type, stores the field (type and data) info
class MeshInstance:
    def __init__(self, type):
        self.field_element_type = {}
        self.type = type
        self.element_attrs = {}
        self.element_data_layout = {}
        self.type_instantiated = False
        self.data_instantiated = False

        self.mesh_ptr = _ti_core.create_mesh()

    def set_owned_offset(self, element_type : MeshElementType, owned_offset : ScalarField):
        _ti_core.set_owned_offset(self.mesh_ptr, element_type, owned_offset.vars[0].ptr.snode())
    
    def set_total_offset(self, element_type : MeshElementType, total_offset : ScalarField):
        _ti_core.set_total_offset(self.mesh_ptr, element_type, total_offset.vars[0].ptr.snode())
    
    def set_num_patches(self, num_patches : int):
        _ti_core.set_num_patches(self.mesh_ptr, num_patches)
    
    def add_mesh_attribute(self, element_type, snode, reordering_type):
        _ti_core.add_mesh_attribute(self.mesh_ptr, element_type, snode, reordering_type)

    def _materialize_fields(self, element_type_set : list, element_num_set : list):
        for element_type, num_elements in zip(element_type_set, element_num_set):
            if self.element_data_layout[element_type] is impl.SOA:
                for attr in self.element_attrs[element_type]:
                    root.dense(indices(0), num_elements).place(attr.field)
            else:
                fields = []
                for attr in self.element_attrs[element_type]:
                    fields.appends(attr.field)
                if len(fields) > 0:
                    root.dense(indices(0), num_elements).place(fields)
        
            for attr in self.element_attrs[element_type]:
                for i in range(len(attr.field.vars)):
                    self.add_mesh_attribute(element_type, attr.field.vars[i].ptr.snode(), attr.reordering)
        
        self.data_instantiated = True
    
    def _type_but_no_data_check(self):
        if not self.type_instantiated:
            raise RuntimeError(f'The type info of this mesh should be instantiated first')
        if self.data_instantiated:
            raise RuntimeError(f'This mesh has been instantiated already!')

class TriInstance(MeshInstance):
    def __init__(self, type):
        super().__init__(type)
        self.relations = MeshGlobalRelationGroup(MeshElementType.Face)
    
    def load_from_obj(self, filename : str, swapyz=False):
        self._type_but_no_data_check()

        obj = OBJFile(filename, swapyz)

        # Compute edges
        edges_dist = {}
        edge_vertex = []
        face_edge = []

        def add_edge(e):
            if e not in edges_dist:
                num_edges = len(edge_vertex)
                edges_dist[e] = num_edges
                edge_vertex.append(e)

        for face in obj.faces:
            e0 = (min(face[0], face[1]), max(face[0], face[1]))
            e1 = (min(face[1], face[2]), max(face[1], face[2]))
            e2 = (min(face[2], face[0]), max(face[2], face[0]))
            add_edge(e0)
            add_edge(e1)
            add_edge(e2)

            face_edge.append([edges_dist[e0], edges_dist[e1], edges_dist[e2]])
        
        num_vertices = obj.vertices.shape[0]
        num_edges = len(edge_vertex)
        num_faces = obj.faces.shape[0]

        self._materialize_fields([MeshElementType.Vertex, MeshElementType.Edge, MeshElementType.Face], 
                                 [num_vertices, num_edges, num_faces])

        # Load positions
        assert hasattr(self, "pos"), "Mesh must have \"pos\" attribute"
        self.pos.from_numpy(obj.vertices)

        # Load global relations
        self.relations.relation(MeshRelationType.FV).validate(num_faces).from_numpy(obj.faces)
        self.relations.relation(MeshRelationType.FE).validate(num_faces).from_numpy(np.asarray(face_edge, dtype=np.uint32))
        self.relations.relation(MeshRelationType.EV).validate(num_edges).from_numpy(np.asarray(edge_vertex, dtype=np.uint32))

        self.data_instantiated = True
        return self

class TetInstance(MeshInstance):
    def __init__(self, type):
        super().__init__(type)
        self.relations = MeshGlobalRelationGroup(MeshElementType.Cell)

    def load_from_file(self, filename):
        self._type_but_no_data_check()

        # TODO(changyu): Impl tet part
        file = TetFile(filename)

        # Compute edges/faces
        edges_dist = {}
        faces_dist = {}
        edge_vertex = []
        cell_edge = []
        face_vertex = []
        cell_face = []

        def add_edge(e):
            if e not in edges_dist:
                num_edges = len(edge_vertex)
                edges_dist[e] = num_edges
                edge_vertex.append(e)
        
        def add_face(f):
            if f not in faces_dist:
                num_faces = len(face_vertex)
                faces_dist[f] = num_faces
                face_vertex.append(f)
        
        def sort3(a, b, c):
            if a > c: a, c = c, a
            if b > c: b, c = c, b
            if a > b: a, b = b, a
            return (a, b, c)

        for cell in file.cells:
            e0 = (min(cell[0], cell[1]), max(cell[0], cell[1]))
            e1 = (min(cell[1], cell[2]), max(cell[1], cell[2]))
            e2 = (min(cell[2], cell[3]), max(cell[2], cell[3]))
            e3 = (min(cell[3], cell[0]), max(cell[3], cell[0]))
            e4 = (min(cell[0], cell[2]), max(cell[0], cell[2]))
            e5 = (min(cell[1], cell[3]), max(cell[1], cell[3]))
            
            f0 = sort3(cell[0], cell[1], cell[2])
            f1 = sort3(cell[0], cell[1], cell[3])
            f2 = sort3(cell[0], cell[2], cell[3])
            f3 = sort3(cell[1], cell[2], cell[3])

            add_edge(e0)
            add_edge(e1)
            add_edge(e2)
            add_edge(e3)
            add_edge(e4)
            add_edge(e5)

            add_face(f0)
            add_face(f1)
            add_face(f2)
            add_face(f3)

            cell_edge.append([edges_dist[e0], edges_dist[e1], edges_dist[e2], edges_dist[e3], edges_dist[e4], edges_dist[e5]])
            cell_face.append([faces_dist[f0], faces_dist[f1], faces_dist[f2], faces_dist[f3]])

        num_vertices = file.vertices.shape[0]
        num_edges = len(edges_dist)
        num_faces = len(faces_dist)
        num_cells = file.cells.shape[0]

        self._materialize_fields([MeshElementType.Vertex, MeshElementType.Edge, MeshElementType.Face, MeshElementType.Cell], 
                                 [num_vertices, num_edges, num_faces, num_cells])

        # Load positions
        assert hasattr(self, "pos"), "Mesh must have \"pos\" attribute"
        self.pos.from_numpy(file.vertices)

        # Load global relations
        self.relations.relation(MeshRelationType.CV).validate(num_cells).from_numpy(file.cells)
        self.relations.relation(MeshRelationType.CF).validate(num_cells).from_numpy(np.asarray(cell_face, dtype=np.uint32))
        self.relations.relation(MeshRelationType.CE).validate(num_cells).from_numpy(np.asarray(cell_edge, dtype=np.uint32))
        self.relations.relation(MeshRelationType.FV).validate(num_faces).from_numpy(np.asarray(face_vertex, dtype=np.uint32))
        self.relations.relation(MeshRelationType.EV).validate(num_edges).from_numpy(np.asarray(edge_vertex, dtype=np.uint32))

        self.data_instantiated = True
        return self

# Define the Mesh Type, stores the field type info
class MeshType:
    def __init__(self):
        pass

class TriType(MeshType):
    def __init__(self):
        self.vertices = MeshElement(MeshElementType.Vertex)
        self.edges = MeshElement(MeshElementType.Edge)
        self.faces = MeshElement(MeshElementType.Face)

    def instance(self):
        inst = TriInstance(self)
        for element_type, data in zip([MeshElementType.Vertex, MeshElementType.Edge, MeshElementType.Face],
                                [self.vertices, self.edges, self.faces]):
            inst.element_data_layout[element_type] = data.data_layout
            inst.element_attrs[element_type] = data.instance(inst, element_type)
        
        inst.type_instantiated = True
        return inst

class TetType(MeshType):
    def __init__(self):
        self.vertices = MeshElement(MeshElementType.Vertex)
        self.edges = MeshElement(MeshElementType.Edge)
        self.faces = MeshElement(MeshElementType.Face)
        self.cells = MeshElement(MeshElementType.Cell)
    
    def instance(self):
        inst = TetInstance(self)
        for element_type, data in zip([MeshElementType.Vertex, MeshElementType.Edge, MeshElementType.Face, MeshElementType.Cell],
                                [self.vertices, self.edges, self.faces, self.cells]):
            inst.element_data_layout[element_type] = data.data_layout
            inst.element_attrs[element_type] = data.instance(inst, element_type)

        inst.type_instantiated = True
        return inst
        

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
    def tet():
        return TetType()
    
    @staticmethod
    def tri():
        return TriType()