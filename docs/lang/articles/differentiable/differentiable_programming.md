---
sidebar_position: 1
---

# Differentiable Programming

## Introduction

Differentiable programming proves to be useful in a wide variety of areas
such as scientific computing and artificial intelligence. For instance,
a controller optimization system equipped with differentiable simulators converges one to
four orders of magnitude faster than those using model-free reinforcement learning algorithms.[^1][^2]

[^1]: [End-to-End Differentiable Physics for Learning and Control
](https://papers.nips.cc/paper/2018/file/842424a1d0595b76ec4fa03c46e8d755-Paper.pdf)

[^2]: [ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics](https://arxiv.org/pdf/1810.01054.pdf)

Suppose you have the following kernel:

```python
x = ti.field(float, ())
y = ti.field(float, ())

@ti.kernel
def compute_y():
    y[None] = ti.sin(x[None])
```

Now if you want to get the derivative of `y` with respect to `x`:
`dy/dx`, it is straightforward to write out the gradient kernel manually:

```python {4-6}
x = ti.field(dtype=ti.f32, shape=())
y = ti.field(dtype=ti.f32, shape=())
dy_dx = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def compute_dy_dx():
    dy_dx[None] = ti.cos(x[None])
```

However, as you make a change to `compute_y`, you have to rework the gradient formula
by hand and update `compute_dy_dx` accordingly. Apparently, when
the kernel becomes larger and gets frequently updated, this manual workflow is
really error-prone and hard to maintain.

If you run into this situation, Taichi's handy automatic differentiation (autodiff)
system comes to your rescue! Taichi supports gradient evaluation through
either `ti.ad.Tape()` or the more flexible `kernel.grad()` syntax.

## Using `ti.ad.Tape()`

Let's still take the `compute_y` kernel above for an explanation.
Using `ti.ad.Tape()` is the easiest way to obtain a kernel that computes `dy/dx`:

1.  Enable `needs_grad=True` option when declaring fields involved in
    the derivative chain.
2.  Use context manager `with ti.ad.Tape(y):` to capture the kernel invocations which you want to automatically differentiate.
3.  Now `dy/dx` value at current `x` is available at `x.grad[None]`.

The following code snippet explains the steps above:

```python
x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_y():
    y[None] = ti.sin(x[None])

with ti.ad.Tape(y):
    compute_y()

print('dy/dx =', x.grad[None], ' at x =', x[None])
```
### Case study: gravity simulation

A common problem in physical simulation is that it is usually easy to compute
energy but hard to compute force on every particle,
for example [Bond bending (and torsion) in molecular dynamics](https://github.com/victoriacity/taichimd/blob/5a44841cc8dfe5eb97de51f1d46f1bede1cc9936/taichimd/interaction.py#L190-L220)
and [FEM with hyperelastic energy functions](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fem128.py).
Recall that we can differentiate
(negative) potential energy to get forces: `F_i = -dU / dx_i`. So once you have coded
a kernel that computes the potential energy, you may use Taichi's autodiff
system to obtain the derivatives and then `F_i` on each particle.

Taking
[examples/simulation/ad_gravity.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/ad_gravity.py)
as an example:

```python
import taichi as ti
ti.init()

N = 8
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy


@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]  # dx/dt = v


def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        compute_U(
        )  # The tape will automatically compute dU/dx and save the results in x.grad
    advance()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]


init()
gui = ti.GUI('Autodiff gravity')
while gui.running:
    for i in range(50):
        substep()
    gui.circles(x.to_numpy(), radius=3)
    gui.show()
```

:::note

The argument `U` to `ti.ad.Tape(U)` must be a 0D field.

To use autodiff with multiple output variables, see the
`kernel.grad()` usage below.
:::

:::note

`ti.ad.Tape(U)` automatically sets _`U[None]`_ to `0` on
start up.
:::

:::tip
See
[examples/simulation/mpm_lagrangian_forces.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm_lagrangian_forces.py)
and
[examples/simulation/fem99.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fem99.py)
for examples on using autodiff-based force evaluation MPM and FEM.
:::

## Using `kernel.grad()`

As mentioned above, `ti.ad.Tape()` can only track a 0D field as the output variable.
If there are multiple output variables that you want to back-propagate
gradients to inputs, call `kernel.grad()` instead of `ti.ad.Tape()`.
Different from using `ti.ad.Tape()`, you need to set the `grad` of the output variables themselves to `1` manually
before calling `kernel.grad()`. The reason is that the `grad` of the output variables themselves
will always be multiplied to the `grad` with respect to the inputs at the end of the back-propagation.
By calling `ti.ad.Tape()`, you have the program do this under the hood.

```python {13-14}
import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func():
    for i in x:
       loss[None] += x[i] ** 2
       loss2[None] += x[i]

for i in range(N):
    x[i] = i

# Set the `grad` of the output variables to `1` before calling `func.grad()`.
loss.grad[None] = 1
loss2.grad[None] = 1

func()
func.grad()
for i in range(N):
    assert x.grad[i] == i * 2 + 1
```

:::tip
It may be tedius to write out `need_grad=True` for every input in a complicated use case.
Alternatively, Taichi provides an API `ti.root.lazy_grad()` that automatically places the
gradient fields following the layout of their primal fields.
:::

:::caution
When using `kernel.grad()`, it is recommended that you always run forward kernel before backward, for example `kernel(); kernel.grad()`. If global fields used in the derivative calculation get mutated in the forward run, skipping
`kernel()` breaks global data access rule #1 below and may produce incorrect gradients.
:::

## Limitations of Taichi autodiff system

Unlike tools such as TensorFlow where **immutable** output buffers are
generated, the **imperative** programming paradigm adopted by Taichi
allows programmers to freely modify global fields.

To make automatic differentiation well-defined under this setting, the following
rules are enforced when writing differentiable programs in Taichi:

### Global Data Access Rules

Currently Taichi's autodiff implementation does not save intermediate results of global fields which might be used in the backward pass. Therefore mutation is forbidden once you've read from a global field.

:::note Global Data Access Rule #1
Once you read an element in a field, the element cannot be mutated anymore.
:::

```python
import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func_broke_rule_1():
    # BAD: broke global data access rule #1, reading global field and before mutation is done.
    loss[None] = x[1] * b[None]
    b[None] += 100


@ti.kernel
def func_equivalent():
    loss[None] = x[1] * 10

for i in range(N):
    x[i] = i
b[None] = 10
loss.grad[None] = 1

with ti.ad.Tape(loss):
    func_broke_rule_1()
# Call func_equivalent to see the correct result
# with ti.ad.Tape(loss):
    # func_equivalent()

assert x.grad[1] == 10.0
```

:::note Global Data Access Rule #2
If a global field element is written more than once, then starting from the second write, the write **must** come in the form of an atomic add ("accumulation", using `ti.atomic_add` or simply `+=`). Although `+=` violates rule #1 above since it reads the old value before computing the sum, it is the only special case of "read before mutation" that Taichi allows in the autodiff system.
:::

```python
import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func_break_rule_2():
    loss[None] += x[1] ** 2
    # Bad: broke global data access rule #2, it's not an atomic_add.
    loss[None] *= x[2]

@ti.kernel
def func_equivalent():
    loss[None] = (2 + x[1] ** 2) * x[2]

for i in range(N):
    x[i] = i
loss.grad[None] = 1
loss[None] = 2

func_break_rule_2()
func_break_rule_2.grad()
# Call func_equivalent to see the correct result
# func_equivalent()
# func_equivalent.grad()
assert x.grad[1] == 4.0
assert x.grad[2] == 3.0
```

### Avoid mixed usage of parallel for-loop and non-for statements

Mixed usage of parallel for-loops and non-for statements are not supported in the autodiff system.
Please split the two kinds of statements into different kernels.

:::note
Kernel body must only consist of either multiple for-loops or non-for statements.
:::

Example:

```python
@ti.kernel
def differentiable_task():
    # Bad: mixed usage of a parallel for-loop and a statement without looping. Please split them into two kernels.
    loss[None] += x[0]
    for i in range(10):
        ...
```

Violation of this rule results in an error.

:::danger DANGER
Violation of rules above might result in incorrect gradient result without a proper error.
We're actively working on improving the error reporting mechanism for it. Please feel free
to open a [github issue](https://github.com/taichi-dev/taichi/issues/new?assignees=&labels=potential+bug&template=bug_report.md&title=)
if you see any silent wrong results.
:::

### Write differentiable code inside a Taichi kernel

Taichi's compiler only captures the code in the Taichi scope when performing the source code transformation for autodiff. Therefore, only the code written in Taichi scope is auto-differentiated. Although you can modify the `grad` of a field in python scope manually, the code is not auto-differentiated.

Example:

```python
import taichi as ti

ti.init()
x = ti.field(dtype=float, shape=(), needs_grad=True)
loss = ti.field(dtype=float, shape=(), needs_grad=True)


@ti.kernel
def differentiable_task():
    for l in range(3):
        loss[None] += ti.sin(x[None]) + 1.0

@ti.kernel
def manipulation_in_kernel():
    loss[None] += ti.sin(x[None]) + 1.0


x[None] = 0.0
with ti.ad.Tape(loss=loss):
    # The line below in python scope only contribute to the forward pass
    # but not the backward pass i.e., not auto-differentiated.
    loss[None] += ti.sin(x[None]) + 1.0

    # Code in Taichi scope i.e. inside Taichi kernels, is auto-differentiated.
    manipulation_in_kernel()
    differentiable_task()

# The outputs are 5.0 and 4.0
print(loss[None], x.grad[None])

# You can modify the grad of a field manually in python scope, e.g., clear the grad.
x.grad[None] = 0.0
# The output is 0.0
print(x.grad[None])
```


## Extending Taichi Autodiff system


Sometimes user may want to override the gradients provided by the Taichi autodiff system. For example, when differentiating a 3D singular value decomposition (SVD) used in an iterative
solver, it is preferred to use a manually engineered SVD derivative subroutine for better numerical stability.
Taichi provides two decorators `ti.ad.grad_replaced` and `ti.ad.grad_for` to overwrite the default
automatic differentiation behavior.


The following is a simple example to use customized gradient function in autodiff:

```
import taichi as ti
ti.init()

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

@ti.ad.grad_replaced
def forward(mul):
    func(mul)
    func(mul)

@ti.ad.grad_for(forward)
def backward(mul):
    func.grad(mul)

with ti.ad.Tape(loss=total):
    forward(4)

assert x.grad[0] == 4
```

Customized gradient function works with both `ti.ad.Tape()` and `kernel.grad()`. More examples can be found at `test_customized_grad.py`.

### Checkpointing

Another use case of customized gradient function is checkpointing. We can use recomputation to save memory space through
a user-defined gradient function.
[diffmpm.py](https://github.com/yuanming-hu/difftaichi/blob/master/examples/diffmpm.py#L226-L244)
demonstrates that by defining a customized gradient function that recomputes the grid states during backward,
we can reuse the grid states and allocate only one copy compared to `O(n)` copies in a native implementation
without customized gradient function.

## DiffTaichi

The [DiffTaichi repo](https://github.com/yuanming-hu/difftaichi)
contains 10 differentiable physical simulators built with Taichi
differentiable programming. A few examples with neural network
controllers optimized using differentiable simulators and brute-force
gradient descent:

![image](https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/ms3_final-cropped.gif)

![image](https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/rb_final2.gif)

![image](https://github.com/yuanming-hu/public_files/raw/master/learning/difftaichi/diffmpm3d.gif)

:::tip
Check out [the DiffTaichi paper](https://arxiv.org/pdf/1910.00935.pdf)
and [video](https://www.youtube.com/watch?v=Z1xvAZve9aE) to learn more
about Taichi differentiable programming.
:::

## Forward-Mode Autodiff

Automatic differentiation (Autodiff) has two modes, reverse mode and forward mode.

- Reverse mode computes Vector-Jacobian Product (VJP), which means computing one *row* of the Jacobian matrix at a time. Therefore, reverse mode is more efficient for functions, which have more inputs than outputs. `ti.ad.Tape()` and `kernel.grad()` are for reverse-mode autodiff.
- Forward mode computes Jacobian-Vector Product (JVP), which means computing one *column* of the Jacobian matrix at a time. Therefore, forward mode is more efficient for functions, which have more outputs than inputs. As of v1.1.0, Taichi supports forward-mode autodiff. `ti.ad.FwdMode()` and `ti.root.lazy_dual()` are for forward-mode autodiff.

### Using `ti.ad.FwdMode()`

The usage of `ti.ad.FwdMode()` is similar to that of `ti.ad.Tape()`. Here we reuse the example for reverse mode above for `ti.ad.FwdMode()`.

1. Set `needs_dual=True` when declaring fields involved in a derivative chain.

   > The `dual` here indicates `dual number` in math. This is because forward-mode autodiff is equivalent to evaluating a function with dual numbers.

2. Use context manager with `ti.ad.FwdMode(loss=y, param=x)` to capture the kernel invocations to automatically differentiate.

   *Now dy/dx value at the current x is available at function output `y.dual[None]`.*

The following code snippet explains the steps above:

```python
import taichi as ti
ti.init()

x = ti.field(dtype=ti.f32, shape=(), needs_dual=True)
y = ti.field(dtype=ti.f32, shape=(), needs_dual=True)

@ti.kernel
def compute_y():
    y[None] = ti.sin(x[None])

# `loss`: The function's output
# `param`: The input of the function
with ti.ad.FwdMode(loss=y, param=x):
    compute_y()

print('dy/dx =', y.dual[None], ' at x =', x[None])
```

:::note
`ti.ad.FwdMode()` automatically clears the dual field of `loss`.
:::

`ti.ad.FwdMode()` supports multiple inputs and outputs:

- `param` can be an N-D field.
- `loss` can be an individual N-D field or a list of N-D fields.
- `seed` is the 'vector' in Jacobian-vector product, which controls the parameter that is computed derivative with respect to. `seed` is required if `param` is not a scalar field.

The following code snippet shows another two cases with multiple inputs and outputs: With `seed=[1.0, 0.0] `or `seed=[0.0, 1.0]` , we can compute derivatives solely with respect to `x_0` or `x_1`.

```python
import taichi as ti
ti.init()
N_param = 2
N_loss = 5
x = ti.field(dtype=ti.f32, shape=N_param, needs_dual=True)
y = ti.field(dtype=ti.f32, shape=N_loss, needs_dual=True)

@ti.kernel
def compute_y():
    for i in range(N_loss):
        for j in range(N_param):
            y[i] += i * ti.sin(x[j])

# Compute derivatives with respect to x_0
# `seed` is required if `param` is not a scalar field
with ti.ad.FwdMode(loss=y, param=x, seed=[1.0, 0.0]):
    compute_y()
print('dy/dx_0 =', y.dual, ' at x_0 =', x[0])

# Compute derivatives with respect to x_1
# `seed` is required if `param` is not a scalar field
with ti.ad.FwdMode(loss=y, param=x, seed=[0.0, 1.0]):
    compute_y()
print('dy/dx_1 =', y.dual, ' at x_1 =', x[1])
```

:::tip
Just as reverse-mode autodiff, Taichi's forward-mode autodiff provides `ti.root.lazy_dual()`, which automatically places the dual fields following the layout of their primal fields.
:::
