---
sidebar_position: 3
---

# Fields

Fields are global variables provided by Taichi. Currently, it can only be defined before launching any Taichi kernel. Fields can be either
sparse or dense.  An element of a field can be either a scalar or a
vector/matrix. This term is borrowed from mathematics and physics. If you
have already known [scalar field](https://en.wikipedia.org/wiki/Scalar_field) (e.g., heat field), vector field (e.g., [gravitational field](https://en.wikipedia.org/wiki/Gravitational_field)) in mathematics and physics, it would be straightforward to understand the fields in Taichi.

:::note
Matrices can be used as field elements, so you can have fields with each
element being a matrix.
:::

## Scalar fields

A simple example might help you understand scalar fields. Assume you have a rectangular wok on the top of a fire. At each point of the wok, there would be a temperature. The surface of the wok forms a heat field. The width and height of the wok are similar to the `shape` of the Taichi scalar field. The temperature (0-D scalar) is like the element of the Taichi scalar field. We could use the following field to represent the
heat field on the wok:

``` python
heat_field = ti.field(dtype=ti.f32, shape=(width_wok, height_wok))
```

- Every global variable is an N-dimensional field.

  - Global `scalars` are treated as 0-D scalar fields.

- Fields are always accessed by indices

  - E.g. `x[i, j, k]` if `x` is a 3D scalar field.
  - Even when accessing 0-D field `x`, use `x[None] = 0` instead of `x = 0`. Please **always** use indexing to access entries in fields. A 0-D field looks like `energy = ti.field(dtype=ti.f32, shape=())`.
- Field values are initially zero.

- Sparse fields are initially inactive.

## Vector fields
We are all live in a gravitational field which is a vector field. At each position of the 3D space, there is a gravity force vector. The gravitational field could be represent with:
```python
gravitational_field = ti.Vector.field(n = 3,dtype=ti.f32,shape=(x,y,z))
```
`x,y,z` are the sizes of each dimension of the 3D space respectively.  `n` is the number of elements of the gravity force vector.

## Matrix fields

Field elements can also be matrices. In continuum mechanics, each
infinitesimal point in a material exists a strain and a stress tensor. The strain and stress tensor is a 3 by 3 matrix in the 3D space. To represent this tensor field we could use:
```python
strain_tensor_field = ti.Matrix.field(n = 3,m = 3, dtype=ti.f32, shape=(x,y,z))
```

`x,y,z` are the sizes of each dimension of the 3D material respectively. `n, m` are the dimensions of the strain tensor.

In general case, suppose you have a `128 x 64` field called `A`, and each element contains
a `3 x 2` matrix. To allocate a `128 x 64` matrix field which has a
`3 x 2` matrix for each of its entry, use the statement
`A = ti.Matrix.field(3, 2, dtype=ti.f32, shape=(128, 64))`.

- If you want to get the matrix of grid node `i, j`, please use
  `mat = A[i, j]`. `mat` is simply a `3 x 2` matrix.
- To get the element on the first row and second column of that
  matrix, use `mat[0, 1]` or `A[i, j][0, 1]`.
- As you may have noticed, there are **two** indexing operators `[]`
  when you load a matrix element from a global matrix field: the
  first is for field indexing, the second for matrix indexing.
- `ti.Vector` is simply an alias of `ti.Matrix`.

### Matrix size

For performance reasons matrix operations will be unrolled during the compile stage, therefore we
suggest using only small matrices. For example, `2x1`, `3x3`, `4x4`
matrices are fine, yet `32x6` is probably too big as a matrix size.

:::caution
Due to the unrolling mechanisms, operating on large matrices (e.g.
`32x128`) can lead to a very long compilation time and low performance.
:::

If you have a dimension that is too large (e.g. `64`), it's better to
declare a field of size `64`. E.g., instead of declaring
`ti.Matrix.field(64, 32, dtype=ti.f32, shape=(3, 2))`, declare
`ti.Matrix.field(3, 2, dtype=ti.f32, shape=(64, 32))`. Try to put large
dimensions to fields instead of matrices.

## Struct fields
In addition to vectors and matrices, field elements can be user-defined structs. A struct variable may contain scalars, vectors/matrices, or other structs as its members. A struct field is created by providing a dictionary of name and data type of each member. For example, a 1D field of particles with position, velocity, acceleration, and mass for each particle can be represented as:
```python
particle_field = ti.Struct.field({
    "pos": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(3, ti.f32),
    "acc": ti.types.vector(3, ti.f32),
    "mass": ti.f32,
  }, shape=(n,))
```
[Compound types](type.md#compound-types) (`ti.types.vector`, `ti.types.matrix`, and `ti.types.struct`) need to be used to create vectors, matrices, or structs as field members. Apart from using `ti.Struct.field`, the above particle field can be alternatively created using field creation from compound types as:
```python
vec3f = ti.types.vector(3, ti.f32)
particle = ti.types.struct(
  pos=vec3f, vel=vec3f, acc=vec3f, mass=ti.f32,
)
particle_field = particle.field(shape=(n,))
```
Members of a struct field can be accessed either locally (i.e., member of a struct field element) or globally (i.e., member field of a struct field):
```python
# set the position of the first particle to origin
particle_field[0] # local ti.Struct
particle_field[0].pos = ti.Vector([0.0, 0.0, 0.0])

# make the mass of all particles be 1
particle_field.mass # global ti.Vector.field
particle_field.mass.fill(1.0)
```
