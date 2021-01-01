import taichi as ti


@ti.test()
def test_vector():
    type_list = [ti.f32, ti.i32]

    a = ti.Vector.field(len(type_list), dtype=type_list, shape=())
    b = ti.Vector.field(len(type_list), dtype=type_list, shape=())
    c = ti.Vector.field(len(type_list), dtype=type_list, shape=())

    @ti.kernel
    def init():
        a[None] = [1.0, 3]
        b[None] = [2.0, 4]
        c[None] = a[None] + b[None]

    def verify():
        assert isinstance(a[None][0], float)
        assert isinstance(a[None][1], int)
        assert isinstance(b[None][0], float)
        assert isinstance(b[None][1], int)
        assert c[None][0] == 3.0
        assert c[None][1] == 7

    init()
    verify()


@ti.test()
def test_matrix():
    type_list = [[ti.f32, ti.i32], [ti.i64, ti.f32]]
    a = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list, shape=())
    b = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list, shape=()) 
    c = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list, shape=()) 
    
    @ti.kernel
    def init():
        a[None] = [[1.0, 3],[1, 3.0]]
        b[None] = [[2.0, 4],[-2, -3.0]]
        c[None] = a[None] + b[None]

    def verify():
        assert isinstance(a[None][0], float)
        assert isinstance(a[None][1], int)
        assert isinstance(b[None][0], float)
        assert isinstance(b[None][1], int)
        assert c[None][0, 0] == 3.0
        assert c[None][0, 1] == 7
        assert c[None][1, 0] == -1
        assert c[None][1, 1] == 0.0

    init()
    verify()