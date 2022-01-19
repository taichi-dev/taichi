---
sidebar_position: 3
---

# Fields
Taichi fields are used to store data.
In general, fields are used as global data containers that can be read and written from both the Python scope and the Taichi scope.

A field has its own data type and shape and can be considered as a multi-dimensional array of elements.
An element of a field can be a **scalar**, a **vector**, a **matrix**, or a **struct**.
The sparsity of a field element is **dense** by default, but it can also be **sparse**, as detailed in [Sparse computation](/lang/articles/advanced/sparse).

:::note
The term **field** is borrowed from mathematics and physics.
If you have already known [scalar field](https://en.wikipedia.org/wiki/Scalar_field) (e.g., heat field) or vector field (e.g., [gravitational field](https://en.wikipedia.org/wiki/Gravitational_field)) in mathematics and physics,
it will be straightforward to understand the fields in Taichi.
:::

## Scalar fields
We start introducing fields from this very basic type, the elements of scalar fields are simply scalars.
* A 0D scalar field is a single scalar.
* A 1D scalar field is a 1D linear array.
* A 2D scalar field can be used to represent a 2D regular grid of values.
* A 3D scalar field can be used for volumetric data.

### Declaration
``` python
import taichi as ti
ti.init(arch=ti.cpu)

gravity          = ti.field(ti.f32, shape=())           # 0-D
linear_array     = ti.field(ti.i32, shape=128)          # 1-D
gray_scale_image = ti.field(ti.u8,  shape=(640, 480))   # 2-D
volumetric_data  = ti.field(ti.f32, shape=(32, 32, 32)) # 3-D
```

### Access elements of scalar fields
``` python
gravity[None]          = 9.8
linear_array[0]        = 1
gray_scale_image[1,2]  = 255
volumetric_data[3,3,3] = 2.0
```

### Meta data
``` python
linear_array.shape     # (128,)
volumetric_data.dtype  # f32
```

To be noticed:
* Field values are initially zero.
* Fields are **always** accessed by indices. When accessing 0-D field `x`, use `x[None] = 0` instead of `x = 0`.

### Example
An example might help you understand scalar fields.
Assume you have a gray-scale image. At each point in the image, there would be a pixel value. The width and height of the image are similar to the `shape` of the Taichi scalar field. The pixel value (0-D scalar) is like the element of the Taichi scalar field. We could use the following code to generate a gray-scale image with random pixel values:

``` python
import taichi as ti

ti.init(arch=ti.cpu)
width, height = 640,480
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))

@ti.kernel
def fill_image():
    for i,j in gray_scale_image:
      gray_scale_image[i,j] = ti.random()

fill_image()

gui = ti.GUI('gray-scale image with random values', (width, height))
while gui.running:
    gui.set_image(gray_scale_image)
    gui.show()
```

:::tip
In earlier versions of Taichi, you could not allocate new fields after executing the first kernel. Since Taichi v0.8.0, you can use a new class `FieldsBuilder` for dynamic field allocation and destruction. For more details, please see [Field (advanced)](/lang/articles/advanced/layout).
:::

## Vector fields
We are all living in a gravitational field, which is a vector field. At each position in 3D space, there is a gravity force vector. The gravitational field could be represented by:
```python
gravitational_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(x, y, z))
```
`x, y, z` are the sizes of each dimension of the 3D space respectively. `n` is the number of elements of the gravity force vector.

### Access elements of vector fields
There are **two** indexing operators `[]` when you access a member of a vector field: the first is for field indexing, and the second is for vector indexing.
- The gravity force vector could be accessed by `gravitational_field[i, j, k]` (`0 <= i < x, 0 <= j < y, 0 <= k < z`).
- The `p`-th member of the gravity force vector could be accessed by `gravitational_field[i, j, k][p]` (`0 <= p < n`).
- The 0-D vector field `x = ti.Vector.field(n=3, dtype=ti.f32, shape=())` should be accessed by `x[None][p]` (`0 <= p < n`).


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
There are **two** indexing operators `[]` when you access a member of a matrix from a matrix field:
the first is for field indexing, and the second is for matrix indexing.
- If you want to get the matrix of grid node `i, j`, please use `mat = A[i, j]`. `mat` is simply a `3 x 2` matrix.
- To get the element on the first row and second column of that matrix, use `mat[0, 1]` or `A[i, j][0, 1]`.
- The 0-D matrix field `x = ti.Matrix.field(n=3, m=4, dtype=ti.f32, shape=())` should be accessed by `x[None][p, q]` (`0 <= p < n, 0 <= q < m`).
- `ti.Vector` is simply an alias of `ti.Matrix`.

### Matrix size
For performance reasons, matrix operations will be unrolled during the compile stage.
Therefore we suggest using only small matrices. For example, `2x1`, `3x3`, `4x4`
matrices are fine, yet `32x6` is probably too big as a matrix size.

If you have a dimension that is too large (e.g. `64`), it's better to
declare a field of size `64`. E.g., instead of declaring
`ti.Matrix.field(64, 32, dtype=ti.f32, shape=(3, 2))`, declare
`ti.Matrix.field(3, 2, dtype=ti.f32, shape=(64, 32))`. Try to put large
dimensions to fields instead of matrices.

:::caution
Due to the unrolling mechanisms, operating on large matrices (e.g.
`32x128`) can lead to a very long compilation time and low performance.
:::

## Struct fields
Field elements can be user-defined structs.
Struct fields are created by providing the name and data type of each member variable in a dictionary format.
Member variables of struct fields might be scalars, vectors, matrices, or other struct fields.
For example, a 1-D field of particles with position, velocity, acceleration, and mass can be declared as:
```python
particle_field = ti.Struct.field({
    "pos": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(3, ti.f32),
    "acc": ti.types.vector(3, ti.f32),
    "mass": ti.f32,
  }, shape=(n,))
```

[Compound types](type.md#compound-types) (`ti.types.vector`, `ti.types.matrix`, and `ti.types.struct`) are used to declare vectors, matrices, or structs as field members. Apart from using `ti.Struct.field`, the above particle field can also be declared by using the field of compound types:
```python
vec3f = ti.types.vector(3, ti.f32)
particle = ti.types.struct(
  pos=vec3f, vel=vec3f, acc=vec3f, mass=ti.f32,
)
particle_field = particle.field(shape=(n,))
```

Members of a struct field can be accessed either locally (i.e., member of a struct field element)
or globally (i.e., member field of a struct field):
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
