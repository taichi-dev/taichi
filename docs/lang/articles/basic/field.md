---
sidebar_position: 4
---

# Fields

Fields are **global** variables provided by Taichi. **Global** indicates that fields can be read/written from both the Python scope and the Taichi scope. A field can be considered as a multi-dimensional array of elements, and it can be either **dense** or **sparse**. Similar to a NumPy `ndarray` object, a field has a data type and a shape. Moreover, an element of a field can be a scalar, a **vector**, a **matrix**, or a **struct**.

The term **field** is borrowed from mathematics and physics. If you
have already known [scalar field](https://en.wikipedia.org/wiki/Scalar_field) (e.g., heat field) or vector field (e.g., [gravitational field](https://en.wikipedia.org/wiki/Gravitational_field)) in mathematics and physics, it will be straightforward to understand the fields in Taichi.

To be noticed:
* Fields are always accessed by indices.
* Field values are initially zero.
* Sparse fields are initially inactive.

:::tip
In earlier versions of Taichi, you could not allocate new fields after executing the first kernel. Since Taichi v0.8.0, you can use a new class `FieldsBuilder` for dynamic field allocation and destruction. For more details, please see [Field (advanced)](/lang/articles/advanced/layout).
:::

## Scalar fields

A simple example might help you understand scalar fields. Assume you have a rectangular wok on the top of a fire. At each point of the wok, there would be a temperature. The surface of the wok forms a heat field. The width and height of the wok are similar to the `shape` of the Taichi scalar field. The temperature (0-D scalar) is like the element of the Taichi scalar field. We could use the following field to represent the
heat field on the wok:

``` python
heat_field = ti.field(dtype=ti.f32, shape=(width_wok, height_wok))
```

### Access elements of scalar fields
- If `x` is a 3D scalar field (`ti.field(dtype=ti.f32, shape=(10, 20, 30)`), access its element with `x[i, j, k]` (`0 <= i < 10, 0 <= j < 20, 0 <= k < 30`).
- When accessing 0-D field `x`, use `x[None] = 0` instead of `x = 0`. A 0-D field looks like `energy = ti.field(dtype=ti.f32, shape=())`.

:::caution
Please **always** use indexing to access entries in fields.
:::

## Vector fields
We are all live in a gravitational field which is a vector field. At each position of the 3D space, there is a gravity force vector. The gravitational field could be represented with:
```python
gravitational_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y, z))
```
`x, y, z` are the sizes of each dimension of the 3D space respectively. `n` is the number of elements of the gravity force vector.

### Access elements of vector fields
- The gravity force vector could be accessed by `gravitational_field[i, j, k]` (`0 <= i < x, 0 <= j < y, 0 <= k < z`).
- The `p`-th member of the gravity force vector could be accessed by `gravitational_field[i, j, k][p]` (`0 <= p < n`).
- The 0-D vector field `x = ti.Vector.field(n=3, dtype=ti.f32, shape=())` should be accessed by `x[None][p]` (`0 <= p < n`).

:::note
As you may have noticed, there are **two** indexing operators `[]` when you access a member of a vector from a vector field: the first is for field indexing, and the second is for vector indexing.
:::

## Matrix fields

Field elements can also be matrices. In continuum mechanics, each
infinitesimal point in a material exists a strain and a stress tensor. The strain and stress tensor is a 3 by 3 matrix in the 3D space. To represent this tensor field we could use:
```python
strain_tensor_field = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(x, y, z))
```

`x, y, z` are the sizes of each dimension of the 3D material respectively. `n, m` are the dimensions of the strain tensor.

In a general case, suppose you have a `128 x 64` field called `A`, and each element is
a `3 x 2` matrix, you can define it with `A = ti.Matrix.field(3, 2, dtype=ti.f32, shape=(128, 64))`.

### Access elements of matrix fields
- If you want to get the matrix of grid node `i, j`, please use
  `mat = A[i, j]`. `mat` is simply a `3 x 2` matrix.
- To get the element on the first row and second column of that
  matrix, use `mat[0, 1]` or `A[i, j][0, 1]`.
- The 0-D matrix field `x = ti.Matrix.field(n=3, m=4, dtype=ti.f32, shape=())` should be accessed by `x[None][p, q]` (`0 <= p < n, 0 <= q < m`).

:::note
- As you may have noticed, there are **two** indexing operators `[]`
  when you access a member of a matrix from a matrix field: the
  first is for field indexing, and the second is for matrix indexing.
- `ti.Vector` is simply an alias of `ti.Matrix`.
:::

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
In addition to vectors and matrices, field elements can be user-defined structs. A struct variable may contain scalars, vectors/matrices, or other structs as its members. A struct field is created by providing a dictionary of the name and data type of each member. For example, a 1D field of particles with position, velocity, acceleration, and mass for each particle can be represented as:
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

# set the first member of the second position to 1.0
particle_field[1].pos[0] = 1.0

# make the mass of all particles be 1
particle_field.mass # global ti.Vector.field
particle_field.mass.fill(1.0)
```
