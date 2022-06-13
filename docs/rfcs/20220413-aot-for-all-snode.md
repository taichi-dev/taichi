# RFC: AOT for All SNodes

* Author(s): [Ye Kuang](https://github.com/k-ye)
* Date: 2022-04-13
* Relevant Issue: https://github.com/taichi-dev/taichi/issues/4777
---

- [RFC: AOT for All SNodes](#rfc-aot-for-all-snodes)
  - [* Relevant Issue: https://github.com/taichi-dev/taichi/issues/4777](#-relevant-issue-httpsgithubcomtaichi-devtaichiissues4777)
- [TL;DR](#tldr)
- [Background](#background)
- [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Detailed Design](#detailed-design)
  - [A first attempt](#a-first-attempt)
  - [A working design](#a-working-design)
  - [Defining `shape`](#defining-shape)
  - [AoS vs SoA](#aos-vs-soa)
  - [Gradient and AutoDiff](#gradient-and-autodiff)
  - [Python AOT API](#python-aot-api)
  - [C++ AOT API](#c-aot-api)
  - [Backward Compatibility](#backward-compatibility)
- [Alternatives](#alternatives)
- [FAQ](#faq)

# TL;DR

This RFC describes a design and API changes to make AOT support all kinds of SNodes and/or Taichi fields.

# Background

Currently, Taichi fields are defined and used in the following manner:

```py
a = ti.field(ti.i32)
b = ti.field(ti.f32)
ti.root.pointer(ti.ij, 16).dense(ti.ij, 16).place(a, b)

@ti.kernel
def run():
  for I in ti.grouped(a):
    b[I] = a[I] * 4.2
```

While this is convenient for Python users, it imposes challenges for the deployment side.

1. Taichi fields are currently implemented as global variables.

    This would result in the Taichi kernels being "not pure" and relying on implicit information. When saving such kernels into the AOT module, it is also necessary to save all the depdendant global states. Ideally, users should be able to create Taichi fields, and pass them into Taichi kernels as parameters.

2. SNodes types are missing from the AOT module.

    To moving towards the direction of passing fields into kernels, field and SNode types need to be saved into the AOT module as well.

3. Fields data are not managed by the users.

    Because fields are global, the Taichi runtime have to create and manage them. By localizing the fields, decoupling them from Taichi kernels, users can manage the memory resources for these fields.

# Goals

* Provide a SNode API that allows SNodes and Taichi fields to be localized, so that Taichi kernels can be made *pure*.
* Supports describing the complete SNode tree type explicitly.
* Make SNode types serializable into AOT, so that AOT can use all kinds of SNodes.
* The new SNode API should offer compatibility with the existing usage.
* (Uncertain, but highly desired) Decouple the element type from the SNode type, i.e. the situation where matrix fields has to been implemented in the "scattered" way to support SoA layout.

## Non-Goals

* Expand the support for sparse SNodes beyond LLVM's codegen, especially SPIR-V.

# Detailed Design

## A first attempt

This API would clearly allow us to pass in fields into kernels:

```py
a = ti.field(ti.i32)
b = ti.field(ti.f32)
ti.root.pointer(ti.ij, 16).dense(ti.ij, 16).place(a, b)

@ti.kernel
def run(a: ?, b: ?):
  for I in ti.grouped(a):
    b[I] = a[I] * 4.2

run(a, b)
```

However, it doesn't really work for AOT, because `a` and `b` are **attributes of a tree type**. That is, you *cannot* dump `a` and `b`'s types separately.

To understand this problem, we can define something equivalent in C++:

```cpp
struct AB {
  int32_t a;
  float b;
};

using TreeType = PointerDense<AB>;
```

We cannot declare the `run` kernel as `void run(? a, ? b)`. Instead, we have to pass in a `TreeType` instance as a whole into the kernel, i.e., `void run(TreeType &tree)`.

Internally, as you are using Taichi's SNode system to construct hierarchies, you are also constructing a `SNodeTree` type at the same time. This is done by Taichi's [`FieldsBuilder`](https://github.com/taichi-dev/taichi/blob/master/python/taichi/_snode/fields_builder.py).

## A working design

We will make the SNode tree and its type explicit by providing `SNodeTreeBuilder`. Each field needs to be registered into the builder via `add_field()`. `add_field()` does *not* actually do any memory allocation. Instead, it just returns a *field handle*, which can be used to retrieve a field from the tree inside the kernel.

```py
builder = ti.SNodeTreeBuilder()

builder.add_field(dtype=ti.f32, name='x')
builder.add_field(dtype=ti.i32, name='y')
builder.tree()
  .pointer(ti.ij, 4)
  .dense(ti.ij, 5)
  .place('x', 'y')

# `tree_t` stands for "tree type".
tree_t = builder.build()
```

Similarly, `SNodeTreeBuilder.build()` doesn't allocate memory for the tree. It only builds *the type of* a SNode tree. You can later instantiate a tree with `tree_t.instantiate()`. There are a few reasons behind this type-tree decoupling design:

1. We have explicit access to the SNode tree type. This is a must for AOT, but can also be used as type annotations for enhanced language formality.
2. We can instantiate as many trees as we want from this type, and pass them to the same kernel without re-compilation.

Inside a Taichi kernel, the entire tree can be used in the following way:

```py
@ti.kernel
def run(tr: tree_t):
  for I in ti.grouped(tr.x):
    tr.x[I] = tr.y[I] + 2.0

tree = tree_t.instantiate()
run(tree)
```

The only change from the existing API is that, you will need to prepend the fields with `tree.`. The subscript still happens on a field, not the tree (i.e., `tr.x[I]` instead of `tr[I].x`).

There will be two ways to retrieve a field from a tree:

* By name: `add_field()` takes in a `name` parameter. After building a SNode tree, Taichi will generate an attribute for each registered field on that tree. This allows you to directly write `tr.x` to access the field named `'x'`. `name` serves as the unique identifer of the field in the tree. Note that when placing, it is the names being passed in.
* By field handle: You can also use the field handle returned by `add_field()` to access a field. Here's an example:
   ```py
   builder = ti.SNodeTreeBuilder()
   x_handle = builder.add_field(dtype=ti.f32, name='x')
   # boilerplate to generate tree type and instantiate a tree ...

   @ti.kernel
   def foo(tr: tree_t):
     x = ti.static(tr.get_field(x_handle))  # 1
     for i in x:
       x[i] = i * 2.0
   ```

   Note that this design requires that part of the kernel (1) being evaluated inside Python. It also pulls in the global variable `x_handle`, which kind of violates our initial goal. We could require that `x_handle` is passed into the kernel as an argument. But maybe it's fine just to view that as a trivial Python constant?

## Defining `shape`

Like how `ti.field()` works, `add_field` can take in a `shape` parameter. When doing so, the builder will automatically create a new `dense` field under the root of the tree. Note that you should *not* do another place if `shape` is defined.

Here is an example:

```py
builder = ti.SNodeTreeBuilder()

builder.add_field(dtype=ti.f32, name='x', shape=(4, 8))
# This would result an error
# builder.tree().dense(ti.ij, (4, 8)).place('x')
tree_t = builder.build()
```

It is equivalent to this:

```py
builder = ti.SNodeTreeBuilder()

builder.add_field(dtype=ti.f32, name='x')
builder.tree().dense(ti.ij, (4, 8)).place('x')
tree_t = builder.build()
```

## AoS vs SoA

Two composite types require the switch between AoS vs SoA, `ti.Matrix` and `ti.Struct`.

AoS is quite straightforward. One can just use the composite type as the `dtype` of the field. For example:

```py
builder = ti.SNodeTreeBuilder()

builder.add_field(dtype=ti.vec3, name='x')  # ti.vec3 is a vector of 3 ti.f32's
builder.dense(ti.i, 8).place('x')
tree_t = builder.build()
```

For SoA, things get a bit trickier. The **current approach** is to treat each compopnent of the composite type as a standalone scalar Taichi field. In the example below, we have to manually place the underlying 3 components of `x` separately.

```py
# Current way (as of v1.0.1) of doing SoA in Taichi
x = ti.Vector.field(3, ti.f32)
for f in x._get_field_members():  # `x` consists three scalar f32 fields
  ti.root.dense(ti.ij).place(f)
```

This introduces confusion at several spots:

1. Type is not purely decided by `dtype`, but also by how the field is placed.
2. It introduces the notion of "nested field", which Taichi doesn't currently have a good abstraction for. Because of this, it is quite complicated to apply certain kind of optimizations for a composite-typed field. For example, vectorized load/save consumes the same bandwidth as scalar ops on certain platforms. Without a good abstraction, the checking for whether a matrix field is AoS or SoA has to be spread across different passes in CHI IR.

If we further think about the problem, SoA `x` is not really a field. Instead, it is a *grouped view* of three individual scalar fields. This view provides matrix operations, which won't make sense for each individual scalar field.

In addition, because type is currently coupled with Taichi field definition, a Taichi field has to be implemented as individual fields in order to support the SoA scenario. Once we switch to the type builder pattern, we can control how the type is constructed first, and choose the field implementation later.

If we want to make it explicit that this is a *field view*, we can do the following:

```py
builder = ti.SNodeTreeBuilder()
builder.add_field(dtype=ti.f32, name='v0')
builder.add_field(dtype=ti.f32, name='v1')
builder.add_field(dtype=ti.f32, name='v2')
for v in ['v0', 'v1', 'v2']:
  builder.tree().dense(ti.ij, 4).place(v)

# Checks that
# 1. `components` and `dtype` are compatible.
# 2. If `dtype` is a vector/matrix, then all the fields in `components` are homogeneous in their SNode hierarchy.
#    See https://github.com/taichi-dev/taichi/issues/3810
builder.add_field_view(dtype=ti.vec3, name='vel', components=['v0', 'v1', 'v2'])
```

Matrix field view supports common matrix operations, and is equivalent to expanding each component into a local matrix variable.

```py
# 1
vel_soa[i, j].inverse()
# equivalent to
ti.vec3([v0[i, j], v1[i, j], v2[i, j]]).inverse()

# 2
vel_soa[i, j][1] += 2.0
# equivalent to
v1[i, j] += 2.0

# 3
vel_soa[i, j] = vel_soa[i, j] @ some_vec3
# equivalent to
vel_tmp = ti.vec3([v0[i, j], v1[i, j], v2[i, j]])
vel_tmp = vel_tmp @ some_vec3
v0[i, j] = vel_tmp[0]
v1[i, j] = vel_tmp[1]
v2[i, j] = vel_tmp[2]
```

To make field view even more powerful, we can supported *nested field views*. For example:

```py
vertex_t = ti.types.struct({'pos': ti.vec3, 'normal': ti.vec3})
sphere_t = ti.types.struct({'center': vertex_t, 'radius': ti.f32})

builder = ti.SNodeTreeBuilder()
builder.add_field(dtype=ti.vec3, name='pos')
builder.add_field(dtype=ti.vec3, name='normal')
builder.add_field(dtype=ti.f32, name='radius')
builder.add_field_view(dtype=sphere_t, name='spheres',
                       components=[['pos', 'normal'], 'radius'])
###                                 ^^^^^^^^^^^^^^^^^ Note this is nested
```

## Gradient and AutoDiff

In order to support autodiff, `add_field()` still needs to take in a parameter named `needs_grad: bool`:

```py
b = ti.SNodeTreeBuilder()
b.add_field(dtype=ti.f32, name='x', needs_grad=True)
# AOS
b.tree()....place('x', b.grad_of('x'))
# or SOA
b.tree()....place('x')
b.tree()....place(b.grad_of('x'))
```

If `needs_grad=True`, the primal and adjoint fields will be defined inside the same tree. You will need to use `b.grad_of(primal_name)` to access the handle of the adjoint field. The alternative would be to use `f'{primal_name}.grad'`, which feels too ad-hoc.

Alternatively, if you don't want to place the gradient fields on your own, you could use `builder.lazy_grad()` by the end, which automatically places all the gradient fields.

## Python AOT API

Here's the Python AOT API to save the SNodeTree type.

```py
builder = ti.SNodeTreeBuilder()
# ...
tree_t = builder.build()

@ti.kernel
def foo(tr: tree_t):
  # ...

m = ti.aot.Module(arch)
m.add_snode_tree_type(tree_t, name="vel_tree")
m.add_kernel(foo)
m.save('/path/to/module')
```

## C++ AOT API

```cpp
auto mod = taichi::aot::Module("/path/to/module");
auto *tree_t = mod->get_snode_tree("vel_tree");
taichi::Device::AllocParams alloc_params;
alloc_params.size = tree_t->get_size();
auto *tree_mem = device->allocate_memory(alloc_params);
// By doing this, the kernel can verify that the passed in memory matches its
// signature.
auto *tree = taichi::instantiate_tree(tree_t, tree_mem);

auto foo_kernel = mod->get_kernel("foo");
foo_kernel->launch(/*args=*/{tree});
```

## Backward Compatibility

We need to make sure `ti.SNodeTreeBuilder` can still support the existing usage. Right now `ti.root` is already implemented as a "field accumulator": All fields being accumulated in root get materialized into a new SNode tree upon kernel invocation.

Let's start with a simple example:

```py
x = ti.field(ti.f32)
ti.root.pointer(ti.i, 4).dense(ti.i, 8).place(x)

@ti.kernel
def foo():
  for i in x:
    x[i] = i * 2.0
```

The equivalent code using the new `SNodeTreeBuilder` API is shown below:

```py
b = ti.SNodeTreeBuilder()
b.add_field(ti.f32, name='x')
b.tree().pointer(ti.i, 4).dense(ti.i, 8).place('x')
tree_t = b.build()

tr = tree_t.instantiate()

@ti.kernel
def foo():
  for i in tr.x:
    tr.x[i] = i * 2.0
```

In order to provide backward compatibility, we need some helper utils to make the following happen:

* Maps `x@old` to `tr.x@new`. In addition, the runtime will need to know which SNode tree `x@old` belongs to.
* `x@old` returned by `ti.field()` will be a placeholder for field, until the current SNode tree of `ti.root` is built and instantiated.

All these being considered, here's a possible solution.

`ti.root` is just a global `SNodeTreeBuilder`.

`ti.field()` returns a `FieldThunk` ([what is a *thunk*?](https://en.wikipedia.org/wiki/Thunk)).

```py
class FieldThunk:
  def __init__(self, fid):
    self.field_id = fid
    self.tree = None

  def bind(self, tree):
    self.tree = tree

def field(dtype, name='', shape=None, offset=None, needs_grad=False):
  name = name or random_name()
  handle = ti.root.add_field(dtype, name)
  ft = FieldThunk(handle)
  ti.root._field_thunks.append(ft)
  return ft
```

Upon materializing the SNodeTree:

```py
tree_t = ti.root.build()
tree = tree_t.instantiate()
ti._runtime.global_snode_trees.append(tree)
for ft in ti.root._field_thunks:
  ft.bind(tree)

# Make `ti.root` a new SNodeTreeBuilder to allow for dynamic fields
ti.root = SNodeTreeBuilder()
```

When JIT compiling a Taichi kernel, it transforms `x@old` into `x.tree.get_field(x.field_id)`, where `x` is the `FieldThunk`.

# Alternatives

Not sure what's a better design to cover all the listed goals here.

# FAQ

TBD
