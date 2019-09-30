# The Taichi Python Frontend

Make sure you also check out the DiffTaichi paper (section "Language design" and "Appendix A") to learn more about the language.

## Global Tensors
 - Every global variable is an N-dimensional tensor. Global scalars are treated as 0-D tensors.
 - Global tensors are accessed using indices, e.g. `x[i, j, k]` if `x` is a 3D tensor. For 0-D tensor, access it as `x[None]`.
   - If you access a 0-D tensor `x` using `x = 0`, instead of `x[None] = 0`, the handle `x` will be set to zero instead of the value in that tensor. This is a compromise to the native python semantics. So please always use indexing to access entries in tensors.
 - For a tensor `F` of element `ti.Matrix`, make sure you first index the tensor dimensions, and then the matrix dimensions: `F[i, j, k][0, 2]`. (Assuming `F` is a 3D tensor with `ti.Matrix` of size `3x3` as element)
 - `ti.Vector` is simply an alias of `ti.Matrix`.
 - Tensors values are initially zero.
 - Sparse tensors are initially inactive.

## Defining your kernels
 - Kernel arguments must be type hinted. Kernels can have at most 8 scalar parameters, e.g.
```python
@ti.kernel
def print(x: ti.i32, y: ti.f32):
  ti.print(x + y)
```
 - Restart the Taichi runtime system (clear memory, desctory all variables and kernels) : `ti.reset()`
 - Right now kernels can have either statements or at most one for loop.
```python
# Good kernels
@ti.kernel
def print(x: ti.i32, y: ti.f32):
  ti.print(x + y)
  ti.print(x * y)

@ti.kernel
def copy():
  for i in x:
    y[i] = x[i]

# Bad kernels that won't compile right now.
# (split them into two kernels for now. Compiler support coming soon.)
@ti.kernel
def print():
  ti.print(x + y)
  for i in x:
    y[i] = x[i]

@ti.kernel
def print():
  for i in x:
    y[i] = x[i]
  for i in x:
    z[i] = x[i]
```

- `Taichi`-scope (`ti.kernel`) v.s. `Python`-scope: everything decorated by `ti.kernel` is in `Taichi`-scope, which will be compiled by the Taichi compiler.
 - TODO: "It would be useful if there was a way to pass in global vars to kernels. At the moment I'm duplicating code in 2 kernels, each of which operates on a different global var."
## Functions
 - Use `@ti.func` to decorate your Taichi functions. These functions are callable only in `Taichi`-scope. Don't call them in `Python`-scope. All function calls are force-inlined, so no recursion supported.
```python
@ti.func
def laplacian(t, i, j):
  return inv_dx2 * (
      -4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] + p[t, i + 1, j] +
      p[t, i - 1, j])

@ti.kernel
def fdtd(t: ti.i32):
  for i in range(n_grid): # Parallelized over GPU threads
    for j in range(n_grid):
      laplacian_p = laplacian(t - 2, i, j)
      laplacian_q = laplacian(t - 1, i, j)
      p[t, i, j] = 2 * p[t - 1, i, j] + (
          c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                     t - 2, i, j] - c * alpha * dt * laplacian_p
```
 - Functions with multiple return values are not supported now. Use a local variable instead:
```python
# Good function
@ti.func
def safe_sqrt(x):
  rst = 0.0
  if x >= 0:
    rst = ti.sqrt(x)
  else:
    rst = 0.0
  return rst

# Bad function with two *return*s
@ti.func
def safe_sqrt(x):
  if x >= 0:
    return ti.sqrt(x)
  else:
    return 0.0
```


# Data Layout
 - Non-power-of-two tensor dimensions are promoted into powers of two. For example, a tensor of size `(18, 65)` will be materialized as `(32, 128)`. Be careful if you want to iterate over this structural node when it is dense: the loop variables will become iterate over the promoted large domain instead of the original compact domain. Use a range-for instead. For sparse structural nodes, this makes no difference.

# Scalar, Vector and Matrix Arithmetics
 - Supported data types: `ti.i32`, `ti.i64`, `ti.f32`, `ti.f64`.
 - Binary operations on different types will give you a promoted type, e.g. `i32 + f32 = f32`.
 - `ti.Matrix` are for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D global tensor. `ti.Vector` is the same as `ti.Matrix`, except that it has only 1 column.
 - **Differentiate element-wise product `*` and matrix product `@`**.
 - Supported scalar functions:
   - `ti.sin(x)`
   - `ti.cos(x)`
   - `ti.cast(x, type)`
   - `ti.sqr(x)`
   - `ti.floor(x)`
   - `ti.inv(x)`
   - `ti.tan(x)`
   - `ti.tanh(x)`
   - `ti.exp(x)`
   - `ti.log(x)`
   - `ti.abs(x)`
   - `ti.random(type)`
   - `ti.max(a, b)` Note: do not use native python `max` in Taichi kernels.
   - `ti.min(a, b)` Note: do not use native python `min` in Taichi kernels.
   - `ti.length(dynamic_snode)`

# Debugging
 - Debug your program with `ti.print(x)`.

# Performance tips
## Avoid synchronization
 - When using GPU, an asynchronous task queue will be maintained. Whenever reading/writing global tensors, a synchronization will be invoked, which leads to idle cycles on CPU/GPU.
## Make Use of GPU Shared Memory and L1-d$
 -
## Vectorization and Parallelization on CPUs
 -

## Tweaking your data structure
### Improve Cacheline Utilization
### Reduce Data Structure Overheads



# Differentiable Programming on Sparse Data
Recall that a (primal) kernel (e.g. `f(X, Y) = Z, W`) takes as input multiple sparse tensors (e.g. `X, Y`) and outputs another set of sparse tensors (e.g. `Z, W`).  A computation task typically involves a series of kernels. When it comes to differentiable programming, a loss funtion will be defined on the final tensors. The gradients of the final loss function `L` w.r.t. each tensors are computed. In our system, all tensors can be **sparse**.

The **adjoint tensor** of `X_ijk`, `X*_ijk` has entries `X*_ijk = dL / dXijk`.  At a high level, our automatic differentiation (AD) system transforms a **primal** kernel (`f(X, Y) = Z, W`) into its **adjoint** form (`f*(X, Y, Z*, W*) = X*, Y*`).

## The `make_adjoint` Pass (Reverse-Mode Auto-Differentiation)
 - This pass transforms a forward evaluation (primal) kernel into its gradient accumulation (adjoint) kernel.
 - It operates on the hierarchical intermediate representation (IR) of Taichi kernels.
 - Before the `make_adjoint` pass, the simplification pass will simplify most branching into `select(cond, x, y)`, so `make_adjoint` basically takes straight-line code. Even nested `if` statements will be transformed into `select`'s.
 - An outer parallel for-loop is allowed for the primal kernel. The Taichi compiler will distribute these parallel iterations onto CPU/GPU threads.
 - More details on this transform: For each statement, an adjoint variable will be allocated for contribution accumulation. The compiler will traverse the statements in **reverse** order, and accumulates the gradients to the corresponding adjoint variable. An example:
```
for i in range(0, 16, step 1) {
  %1 = load x[i]
  %2 = mul %1, %1
  %3 = sin(%2)
  store y[i] = %3
}
```
will be transformed into
```
for i in range(0, 16, step 1) {
  // adjoint variables
  %1adj = alloca 0.0
  %2adj = alloca 0.0
  %3adj = alloca 0.0
  // original forward computation
  %1 = load x[i]
  %2 = mul %1, %1
  %3 = sin(%2)
  // reverse accumulation
  %4 = load y_adj[i]
  %3adj += %4
  %5 = cos(%2)
  %2adj += %3adj * %5
  %1adj += 2 * %1 * %2adj
  store x_adj[i] = %1adj
}
```


## Comparison with other AD libraries such as TensorFlow/PyTorch
**In one word: higher performance on sparse data and algorithms.** In detail:
 - **Sparsity**. The first-class sparse data structure primitives make it possible to develop efficient systems on sparse data, such as point clouds and voxels. This makes the system especially favorable in 3D computer vision tasks where data (e.g LIDAR scans) are typically sparse.
 - **Imperative**. TensorFlow 1.0 represents computation as a fixed functional computational graph. In our system, the computation is completely imperative and dynamic, which is closer to PyTorch and TensorFlow 2.0.
 - **Finer Granularity**. Those NN libraries often only expose users with basic large-grained operations like `conv`, `batch_norm` and `relu`, namely the neural network layers. Although it is not impossible to compose these basic operations into complex ones, the resulted performance is often suboptimal. This is because producer/consumer locality is not utilized, since compiler optimization **does not optimize across these operations**. Our system exposes fine-grained `instructions` to users and an effective optimizing compiler will deliver high-performance code.
 - **Lower Overhead**. With TensorFlow or PyTorch, since operations typically operate on big data blobs (e.g. `128x27x27x128xfloat32`), the high runtime overhead (e.g. scheduling, kernel launches on GPU) can be amortized over data entries. Our system has lower overhead, making it suitable for sparse tensor processing where there is no sufficient data to amortize the overhead.
 - **More Control**. The user takes care of memory allocation and gradient evaluation, and is in charge of invoking the gradient kernels in the correct order. The flexibility allows easier checkpointing to trade time for space complexity, which is critical in many applications. Programmers have to do a little more work in our system though, but the performance gain makes extra work worthy.

## Note
 - No gradients are propagated to `int` tensors/locals
 - Remember to place your grad tensors, or use `ti.root.lazy_grad()`
 - The user should make sure `grad` tensors have the same sparsity as the corresponding `primal` tensors.
 - Reset gradients every time.

# Multi-Stage Programming
 - Use `ti.static` for compile-time branching (For those who come from C++17, this is [`if constexpr`](https://en.cppreference.com/w/cpp/language/if).)
```python
enable_projection = True

@ti.kernel
def static():
  if ti.static(enable_projection): # No runtime overhead
    x[0] = 1
```
 - Use `ti.static` for forced loop unrolling
```python
@ti.kernel
def g2p(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, real)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
         0.5 * ti.sqr(fx - 0.5)]
    new_v = ti.Vector([0.0, 0.0])
    new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

    # Unrolled 9 iterations for higher performance
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.cast(ti.Vector([i, j]), real) - fx
        g_v = grid_v_out[base(0) + i, base(1) + j]
        weight = w[i](0) * w[j](1)
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[f + 1, p] = new_v
    x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
    C[f + 1, p] = new_C
```
 - Parameterize kernels with different global variables:
```python
import taichi_lang as ti

x = ti.global_var(ti.f32)
y = ti.global_var(ti.f32)
z = ti.global_var(ti.f32)
loss = ti.global_var(ti.f32)

@ti.layout
def tensors():
  ti.root.dense(ti.i, 16).place(x, y, z)
  ti.root.place(loss)
  ti.root.lazy_grad()
  
# Define a function that takes global variables
# ... and returns a materialize kernel
def double(a, b):
  @ti.kernel
  def kernel():
    for i in range(16):
      b[i] = a[i] * 2 + 1
  # Make sure you materialize the kernels immediately
  # (by default they are initialized on first invocation)
  kernel.materialize()
  kernel.grad.materialize() # If you need the gradients
  return kernel
  
@ti.kernel
def compute_loss():
  for i in range(16):
    ti.atomic_add(loss, z[i])
      
for i in range(16):
  x[i] = i
  
# Instantiate your kernels here with different global variables
double1 = double(x, y)
double2 = double(y, z)
with ti.Tape(loss):
  double1()
  double2()
  compute_loss()

for i in range(16):
  print(z[i], x.grad[i])
```
# Python Frontend
Embedding the language in `python` has the following advantages:
- Easy to learn. Python itself is very easy to learn, so is PyTaichiLang.
- Easy to run. No ahead-of-time compilation is needed.
- It allows people to reuse existing python infrastructure:
   - IDEs. A python IDE simply works for TaichiLang, with syntax highlighting, checking, and autocomplete.
   - Package manager (pip). A developed Taichi application and be easily submitted to `PyPI` and others can easily set it up   with `pip`.
   - Existing packages. Interacting with other python components is just trivial.
- The built-in AST manipulation tools in `python` allow us to do magical things, as long as the kernel body can be parsed by the `python` parser.

However, this design decision has drawbacks as well:
 - Indexing is always needed when accessing elements in tensors, even if the tensor is 0D. Use `x[None] = 123` to set the value in `x` if `x` is 0D. This is because `x = 123` will set `x` itself (instead of its containing value) to be the constant `123` in python syntax, and unfortunately we cannot modify this behavior.
 - When dealing with local matrices, syntax like `x(0, 1).val = y` is needed. It would be ideal to write `x(0, 1) = y`, but in `python` assigning to function call is not allowed. For global matrices you can use `x(0, 1)[i, j, k] = 42` and no special attention is needed.
