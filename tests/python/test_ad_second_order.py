import taichi as ti

ti.init(print_ir=True)

x = ti.field(float, shape=(), needs_grad=True)
x_2 = ti.field(float, shape=(), needs_grad=True)
x_3 = ti.field(float, shape=(), needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)


@ti.kernel
def test():
    loss[None] += x[None]**3 + x[None]**2 + ti.sin(x[None])

x[None] = 3.1415926 / 6
x.dual[None] = 1.0
x.adjoint[None] = 0.0
loss.adjoint[None] = 1.0
loss.dual[None] = 0.0
test()
test.grad()
print(x.adjoint[None], x.dual[None] - 1.0)
print(loss.adjoint[None], loss.dual[None])

computed_derivative = x.dual[None] - 1.0
analytical_derivative = 6 * x[None] + 2 - ti.sin(x[None])
assert computed_derivative - analytical_derivative < 1e-6