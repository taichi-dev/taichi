import taichi as ti


@ti.all_archs
def test_complex_kernels():
    x = ti.field(ti.f32)
    total = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(total)
    ti.root.lazy_grad()

    @ti.kernel
    def func(mul: ti.f32):
        for i in range(n):
            ti.atomic_add(total[None], x[i] * mul)

    @ti.complex_kernel
    def forward(mul):
        func(mul)
        func(mul)

    @ti.complex_kernel_grad(forward)
    def backward(mul):
        func.grad(mul)

    with ti.Tape(loss=total):
        forward(4)
    for i in range(n):
        assert x.grad[0] == 4


@ti.all_archs
def test_complex_kernels_indirect():
    x = ti.field(ti.f32)
    total = ti.field(ti.f32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(total)
    ti.root.lazy_grad()

    @ti.kernel
    def func(mul: ti.f32):
        for i in range(n):
            ti.atomic_add(total[None], x[i] * mul)

    def func_proxy(mul):
        func(mul)

    @ti.complex_kernel
    def forward(mul):
        func_proxy(mul)
        func_proxy(mul)

    @ti.complex_kernel_grad(forward)
    def backward(mul):
        func.grad(mul)

    with ti.Tape(loss=total):
        forward(4)
    for i in range(n):
        assert x.grad[0] == 4


@ti.all_archs
def test_complex_kernels_oop():
    @ti.data_oriented
    class A:
        def __init__(self):
            self.x = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.n = 128

            ti.root.dense(ti.i, self.n).place(self.x)
            ti.root.place(self.total)

        @ti.kernel
        def func(self, mul: ti.f32):
            for i in range(self.n):
                ti.atomic_add(self.total[None], self.x[i] * mul)

        @ti.complex_kernel
        def forward(self, mul):
            self.func(mul)
            self.func(mul)

        @ti.complex_kernel_grad(forward)
        def backward(self, mul):
            self.func.grad(mul)

    a = A()

    ti.root.lazy_grad()

    with ti.Tape(loss=a.total):
        a.forward(4)
    for i in range(a.n):
        assert a.x.grad[0] == 4


@ti.all_archs
def test_complex_kernels_oop2():
    @ti.data_oriented
    class A:
        def __init__(self):
            self.x = ti.field(ti.f32)
            self.total = ti.field(ti.f32)
            self.n = 128

            ti.root.dense(ti.i, self.n).place(self.x)
            ti.root.place(self.total)

        @ti.kernel
        def func(self, mul: ti.f32):
            for i in range(self.n):
                ti.atomic_add(self.total[None], self.x[i] * mul)

        def func_proxy(self, mul):
            self.func(mul)

        @ti.complex_kernel
        def forward(self, mul):
            self.func_proxy(mul)
            self.func_proxy(mul)

        @ti.complex_kernel_grad(forward)
        def backward(self, mul):
            self.func.grad(mul)

    a = A()

    ti.root.lazy_grad()

    with ti.Tape(loss=a.total):
        a.forward(4)
    for i in range(a.n):
        assert a.x.grad[0] == 4
