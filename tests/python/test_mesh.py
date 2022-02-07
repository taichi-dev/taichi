import os

import numpy as np
from taichi.lang.misc import mesh_patch_idx

import taichi as ti

this_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(this_dir, 'ell.json')


@ti.test(require=ti.extension.mesh)
def test_mesh_patch_idx():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'idx': ti.i32})
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))

    @ti.kernel
    def foo():
        for v in model.verts:
            v.idx = mesh_patch_idx()

    foo()
    idx = model.verts.idx.to_numpy()
    assert idx[0] == 6
    assert idx.sum() == 89


def _test_mesh_for(cell_reorder=False, vert_reorder=False, extra_tests=True):
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'t': ti.i32}, reorder=vert_reorder)
    mesh_builder.cells.place({'t': ti.i32}, reorder=cell_reorder)
    mesh_builder.cells.link(mesh_builder.verts)
    mesh_builder.verts.link(mesh_builder.cells)
    mesh_builder.cells.link(mesh_builder.cells)
    mesh_builder.verts.link(mesh_builder.verts)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))

    @ti.kernel
    def cell_vert():
        for c in model.cells:
            for j in range(c.verts.size):
                c.t += c.verts[j].id

    cell_vert()
    total = model.cells.t.to_numpy().sum()
    model.cells.t.fill(0)
    assert total == 892

    @ti.kernel
    def vert_cell():
        for v in model.verts:
            for j in range(v.cells.size):
                v.t += v.cells[j].id

    vert_cell()
    total = model.verts.t.to_numpy().sum()
    model.verts.t.fill(0)
    assert total == 1104

    if not extra_tests:
        return

    @ti.kernel
    def cell_cell():
        for c in model.cells:
            for j in range(c.cells.size):
                c.t += c.cells[j].id

    cell_cell()
    total = model.cells.t.to_numpy().sum()
    model.cells.t.fill(0)
    assert total == 690

    @ti.kernel
    def vert_vert():
        for v in model.verts:
            for j in range(v.verts.size):
                v.t += v.verts[j].id

    vert_vert()
    total = model.verts.t.to_numpy().sum()
    model.verts.t.fill(0)
    assert total == 1144


@ti.test(require=ti.extension.mesh)
def test_mesh_for():
    _test_mesh_for(False, False)
    _test_mesh_for(False, True)


@ti.test(require=ti.extension.mesh, optimize_mesh_reordered_mapping=False)
def test_mesh_reordered_opt():
    _test_mesh_for(True, True, False)


@ti.test(require=ti.extension.mesh, mesh_localize_to_end_mapping=False)
def test_mesh_localize_mapping0():
    _test_mesh_for(False, False, False)
    _test_mesh_for(True, True, False)


@ti.test(require=ti.extension.mesh, mesh_localize_from_end_mapping=True)
def test_mesh_localize_mapping1():
    _test_mesh_for(False, False, False)
    _test_mesh_for(True, True, False)


@ti.test(require=ti.extension.mesh)
def test_mesh_reorder():
    vec3i = ti.types.vector(3, ti.i32)
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'s': ti.i32, 's3': vec3i}, reorder=True)
    mesh_builder.cells.link(mesh_builder.verts)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))

    id2 = np.array([x**2 for x in range(len(model.verts))])
    id123 = np.array([[x**1, x**2, x**3] for x in range(len(model.verts))])
    model.verts.s.from_numpy(id2)
    model.verts.s3.from_numpy(id123)

    @ti.kernel
    def foo():
        for v in model.verts:
            assert v.s == v.id**2
            assert v.s3[0] == v.id**1 and v.s3[1] == v.id**2 and v.s3[
                2] == v.id**3
            v.s = v.id**3
            v.s3 *= v.id

    foo()

    id3 = model.verts.s.to_numpy()
    id234 = model.verts.s3.to_numpy()

    for i in range(len(model.verts)):
        assert model.verts.s[i] == i**3
        assert id3[i] == i**3
        assert model.verts.s3[i][0] == i**2
        assert model.verts.s3[i][1] == i**3
        assert model.verts.s3[i][2] == i**4
        assert id234[i][0] == i**2
        assert id234[i][1] == i**3
        assert id234[i][2] == i**4


@ti.test(require=ti.extension.mesh)
def test_mesh_minor_relations():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'y': ti.i32})
    mesh_builder.edges.place({'x': ti.i32})
    mesh_builder.cells.link(mesh_builder.edges)
    mesh_builder.verts.link(mesh_builder.cells)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))
    model.edges.x.fill(1)

    @ti.kernel
    def foo():
        for v in model.verts:
            for i in range(v.cells.size):
                c = v.cells[i]
                for j in range(c.edges.size):
                    e = c.edges[j]
                    v.y += e.x

    foo()
    total = model.verts.y.to_numpy().sum()
    assert total == 576


@ti.test(require=ti.extension.mesh, demote_no_access_mesh_fors=True)
def test_multiple_meshes():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'y': ti.i32})
    meta = ti.Mesh.load_meta(model_file_path)
    model1 = mesh_builder.build(meta)
    model2 = mesh_builder.build(meta)

    model1.verts.y.from_numpy(
        np.array([x**2 for x in range(len(model1.verts))]))

    @ti.kernel
    def foo():
        for v in model1.verts:
            model2.verts.y[v.id] = v.y

    foo()
    out = model2.verts.y.to_numpy()
    for i in range(len(out)):
        assert out[i] == i**2


@ti.test(require=ti.extension.mesh)
def test_mesh_local():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'a': ti.i32})
    mesh_builder.faces.link(mesh_builder.verts)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))
    ext_a = ti.field(ti.i32, shape=len(model.verts))

    @ti.kernel
    def foo(cache: ti.template()):
        if ti.static(cache):
            ti.mesh_local(ext_a, model.verts.a)
        for f in model.faces:
            m = f.verts[0].id + f.verts[1].id + f.verts[2].id
            f.verts[0].a += m
            f.verts[1].a += m
            f.verts[2].a += m
            ext_a[f.verts[0].id] += m
            ext_a[f.verts[1].id] += m
            ext_a[f.verts[2].id] += m

    foo(False)
    res1 = model.verts.a.to_numpy()
    res2 = ext_a.to_numpy()
    model.verts.a.fill(0)
    ext_a.fill(0)
    foo(True)
    res3 = model.verts.a.to_numpy()
    res4 = ext_a.to_numpy()

    for i in range(len(model.verts)):
        assert res1[i] == res2[i]
        assert res1[i] == res3[i]
        assert res1[i] == res4[i]


@ti.test(require=ti.extension.mesh, experimental_auto_mesh_local=True)
def test_auto_mesh_local():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.verts.place({'a': ti.i32, 's': ti.i32})
    mesh_builder.faces.link(mesh_builder.verts)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))
    ext_a = ti.field(ti.i32, shape=len(model.verts))

    @ti.kernel
    def foo(cache: ti.template()):
        for v in model.verts:
            v.s = v.id
        if ti.static(cache):
            ti.mesh_local(ext_a, model.verts.a)
        for f in model.faces:
            m = f.verts[0].s + f.verts[1].s + f.verts[2].s
            f.verts[0].a += m
            f.verts[1].a += m
            f.verts[2].a += m
            for i in range(3):
                ext_a[f.verts[i].id] += m

    foo(False)
    res1 = model.verts.a.to_numpy()
    res2 = ext_a.to_numpy()
    model.verts.a.fill(0)
    ext_a.fill(0)
    foo(True)
    res3 = model.verts.a.to_numpy()
    res4 = ext_a.to_numpy()

    for i in range(len(model.verts)):
        assert res1[i] == res2[i]
        assert res1[i] == res3[i]
        assert res1[i] == res4[i]


@ti.test(require=ti.extension.mesh)
def test_nested_mesh_for():
    mesh_builder = ti.Mesh.Tet()
    mesh_builder.faces.place({'a': ti.i32, 'b': ti.i32})
    mesh_builder.faces.link(mesh_builder.verts)
    model = mesh_builder.build(ti.Mesh.load_meta(model_file_path))

    @ti.kernel
    def foo():
        for f in model.faces:
            for i in range(f.verts.size):
                f.a += f.verts[i].id
            for v in f.verts:
                f.b += v.id

    a = model.faces.a.to_numpy()
    b = model.faces.b.to_numpy()
    assert (a == b).all() == 1


@ti.test(require=ti.extension.mesh)
def test_multiple_mesh_major_relations():
    mesh = ti.TetMesh()
    mesh.verts.place({
        's': ti.i32,
        's_': ti.i32,
        's1': ti.i32,
        'a': ti.i32,
        'b': ti.i32,
        'c': ti.i32
    })
    mesh.edges.place({'s2': ti.i32})
    mesh.cells.place({'s3': ti.i32})
    mesh.verts.link(mesh.verts)
    mesh.verts.link(mesh.edges)
    mesh.verts.link(mesh.cells)

    model = mesh.build(ti.Mesh.load_meta(model_file_path))

    @ti.kernel
    def foo():
        for u in model.verts:
            u.s1 = u.id
        for e in model.edges:
            e.s2 = e.id
        for c in model.cells:
            c.s3 = c.id

        ti.mesh_local(model.verts.s1, model.edges.s2, model.cells.s3)
        for u in model.verts:
            a, b, c = 0, 0, 0
            for i in range(u.verts.size):
                a += u.verts[i].s1
            for i in range(u.edges.size):
                b += u.edges[i].s2
            for i in range(u.cells.size):
                c += u.cells[i].s3
            u.s = a * b * c

        for u in model.verts:
            for i in range(u.verts.size):
                u.a += u.verts[i].s1
        for u in model.verts:
            for i in range(u.edges.size):
                u.b += u.edges[i].s2
        for u in model.verts:
            for i in range(u.cells.size):
                u.c += u.cells[i].s3
        for u in model.verts:
            u.s_ = u.a * u.b * u.c

    foo()

    sum1 = model.verts.s.to_numpy().sum()
    sum2 = model.verts.s_.to_numpy().sum()
    assert sum1 == sum2
