import taichi as ti


@ti.host_arch_only
def test_classfunc():
    @ti.data_oriented
    class Array2D:
        def __init__(self, n, m):
            self.n = n
            self.m = m
            self.val = ti.field(ti.f32, shape=(n, m))

        @ti.func
        def inc(self, i, j):
            self.val[i, j] += i * j

        @ti.func
        def mul(self, i, j):
            return i * j

        @ti.kernel
        def fill(self):
            for i, j in self.val:
                self.inc(i, j)
                self.val[i, j] += self.mul(i, j)

    arr = Array2D(128, 128)

    arr.fill()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val[i, j] == i * j * 2


@ti.host_arch_only
def test_oop():
    @ti.data_oriented
    class Array2D:
        def __init__(self, n, m, increment):
            self.n = n
            self.m = m
            self.val = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.increment = increment

            ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
            ti.root.place(self.total)

        @ti.kernel
        def inc(self):
            for i, j in self.val:
                self.val[i, j] += self.increment

        @ti.kernel
        def inc2(self, increment: ti.i32):
            for i, j in self.val:
                self.val[i, j] += increment

        @ti.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * 4

    arr = Array2D(128, 128, 3)

    double_total = ti.field(ti.f32)

    ti.root.place(double_total)
    ti.root.lazy_grad()

    arr.inc()
    arr.inc.grad()
    assert arr.val[3, 4] == 3
    arr.inc2(4)
    assert arr.val[3, 4] == 7

    with ti.Tape(loss=arr.total):
        arr.reduce()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val.grad[i, j] == 4

    @ti.kernel
    def double():
        double_total[None] = 2 * arr.total

    with ti.Tape(loss=double_total):
        arr.reduce()
        double()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val.grad[i, j] == 8


@ti.host_arch_only
def test_oop_two_items():
    @ti.data_oriented
    class Array2D:
        def __init__(self, n, m, increment, multiplier):
            self.n = n
            self.m = m
            self.val = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.increment = increment
            self.multiplier = multiplier
            ti.root.dense(ti.ij, (self.n, self.m)).place(self.val)
            ti.root.place(self.total)

        @ti.kernel
        def inc(self):
            for i, j in self.val:
                self.val[i, j] += self.increment

        @ti.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.multiplier

    arr1_inc, arr1_mult = 3, 4
    arr2_inc, arr2_mult = 6, 8
    arr1 = Array2D(128, 128, arr1_inc, arr1_mult)
    arr2 = Array2D(16, 32, arr2_inc, arr2_mult)

    ti.root.lazy_grad()

    arr1.inc()
    arr1.inc.grad()
    arr2.inc()
    arr2.inc.grad()
    assert arr1.val[3, 4] == arr1_inc
    assert arr2.val[8, 6] == arr2_inc

    with ti.Tape(loss=arr1.total):
        arr1.reduce()
    with ti.Tape(loss=arr2.total, clear_gradients=False):
        arr2.reduce()
    for i in range(arr1.n):
        for j in range(arr1.m):
            assert arr1.val.grad[i, j] == arr1_mult
    for i in range(arr2.n):
        for j in range(arr2.m):
            assert arr2.val.grad[i, j] == arr2_mult


@ti.host_arch_only
def test_oop_inherit_ok():
    # Array1D inherits from object, which makes the callstack being 'class Array2D(object)'
    # instead of '@ti.data_oriented'. Make sure this also works.
    @ti.data_oriented
    class Array1D(object):
        def __init__(self, n, mul):
            self.n = n
            self.val = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.mul = mul
            ti.root.dense(ti.ij, (self.n, )).place(self.val)
            ti.root.place(self.total)

        @ti.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.mul

    arr = Array1D(128, 42)

    ti.root.lazy_grad()

    with ti.Tape(loss=arr.total):
        arr.reduce()
    for i in range(arr.n):
        assert arr.val.grad[i] == 42


@ti.must_throw(ti.KernelDefError)
@ti.host_arch_only
def test_oop_class_must_be_data_oriented():
    class Array1D(object):
        def __init__(self, n, mul):
            self.n = n
            self.val = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.mul = mul
            ti.root.dense(ti.ij, (self.n, )).place(self.val)
            ti.root.place(self.total)

        @ti.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.mul

    arr = Array1D(128, 42)

    ti.root.lazy_grad()

    # Array1D is not properly decorated, this will raise an Exception
    arr.reduce()


@ti.host_arch_only
def test_hook():
    @ti.data_oriented
    class Solver:
        def __init__(self, n, m, hook):
            self.val = ti.field(ti.f32, shape=(n, m))
            self.hook = hook

        def run_hook(self):
            self.hook(self.val)

    @ti.kernel
    def hook(x: ti.template()):
        for i, j in x:
            x[i, j] = 1.0

    solver = Solver(32, 32, hook)
    solver.run_hook()

    for i in range(32):
        for j in range(32):
            assert (solver.val[i, j] == 1.0)
