import taichi as ti


@ti.test(require=ti.extension.sparse)
def test_pointer():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)
    ti.root.place(s)

    s[None] = 0

    @ti.kernel
    def activate():
        ti.activate(ptr, 1)
        ti.activate(ptr, 32)

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    activate()
    func()
    assert s[None] == 32


@ti.test(require=ti.extension.sparse)
def test_non_dfs_snode_order():
    x = ti.field(dtype=ti.i32)
    y = ti.field(dtype=ti.i32)

    grid1 = ti.root.dense(ti.i, 1)
    grid2 = ti.root.dense(ti.i, 1)
    ptr = grid1.pointer(ti.i, 1)
    ptr.place(x)
    grid2.place(y)
    '''
    This SNode tree has node ids that do not follow DFS order:
    S0root
      S1dense
        S3pointer
          S4place<i32>
      S2dense
        S5place<i32>
    '''
    @ti.kernel
    def foo():
        ti.activate(ptr, [0])

    foo()  # Just make sure it doesn't crash
    ti.sync()
