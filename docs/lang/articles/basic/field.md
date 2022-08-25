---
sidebar_position: 1
---

# Fields

The term _field_ is borrowed from mathematics and physics. If you already know [scalar field](https://en.wikipedia.org/wiki/Scalar_field) (for example heat field), or vector field (for example [gravitational field](https://en.wikipedia.org/wiki/Gravitational_field)), then it is straightforward for you to understand fields in Taichi.

Fields in Taichi are the _global_ data containers that can be accessed from both the Python scope and the Taichi scope. Just like an ndarray in NumPy or a tensor in PyTorch, a field in Taichi is defined as a multi-dimensional array of elements, and elements in a field can be a scalar, a vector, a matrix, or a struct.

:::note
A 0D (zero-dimensional) field contains *only* one element.
:::

## Scalar fields

Scalar fields refer to the fields that store scalars and are the most basic fields. A 0D scalar field is a single scalar.


### Declaration

```python
import taichi as ti
ti.init(arch=ti.cpu)

# Declare a 0D scalar field whose data type is f32
f_0d = ti.field(ti.f32, shape=())           # 0D
# Declare a 1D scalar field whose shape is (128)
f_1d = ti.field(ti.i32, shape=128)          # 1D
# Declare a 2D scalar field whose shape is (640, 480)
f_2d = ti.field(ti.u8,  shape=(640, 480))   # 2D
# Declare a 3D scalar field whose data type is f32
f_3d = ti.field(ti.f32, shape=(32, 32, 32)) # 3D
```

### Access elements in a scalar field

The initial value of elements in a scalar filed is zero. Always use explicit indexing to access elements in a scalar field.

:::note
When accessing a 0D field `x`, use `x[None] = 0`, *not* `x = 0`.
:::

```python
# For a 0D field, you are required to use the index None even though it has only one element
f_0d[None] = 10.0
f_1d[0] = 1
f_2d[1, 2] = 255
f_3d[3, 3, 3] = 2.0
```

As mentioned above, you can use a 2D scalar field to represent a 2D grid of values. The following code snippet creates and displays a 640&times;480 image with randomly-generated gray scales:

```python
import taichi as ti
ti.init(arch=ti.cpu)

width, height = 640,480
# Create a 640x480 scalar field, each of its elements representing a pixel value (f32)
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))

@ti.kernel
def fill_image():
  	# Fill the image with random gray
    for i,j in gray_scale_image:
        gray_scale_image[i,j] = ti.random()

fill_image()
# Create a GUI of same size as the gray-scale image
gui = ti.GUI('gray-scale image with random values', (width, height))
while gui.running:
    gui.set_image(gray_scale_image)
    gui.show()
```

:::tip
With Taichi versions earlier than v0.8.0, you cannot allocate new fields after executing a kernel. Starting from v0.8.0, you can use the `FieldsBuilder` class to dynamically allocate or destruct fields. See the [Field (advanced)](../basic/layout.md) for more information.
:::

:::caution WARNING
Taichi does not support slicing on a Taichi field. For example, with the 2D scalar field `f_2d`, you can do `f_2d[1, 2]`, but *not* `f_2d[1]`.
:::

### Metadata

Metadata provides the basic information of a scalar field. You can retrieve the data type and shape of a scalar field via its `shape` and `dtype` property:

```python
f_1d.shape  # (128)
f_3d.dtype  # f32
```

## Vector fields

As the name suggests, vector fields are the fields whose elements are vectors.

- You can use a vector field to represent an RGB image. Then, each of its elements is an (r, g, b) triple.
- You can use a vector field to represent a volumetric field. Then, each of its elements can be the velocity of the corresponding particle.

### Declaration

The following code snippet declares a 3D field of 2D vectors:

```python
# Declare a 1x2x3 vector field, whose vector dimension is n=2
f = ti.Vector(ti.f32, n=2).field(shape=(1,2,3))
```

The following code snippet declares a `300x300x300` vector field `volumetric_field`, whose vector dimension is 3:

```python
box_size = (300, 300, 300)  # A 300x300x300 grid in a 3D space
# Declare a 300x300x300 vector field, whose vector dimension is n=3
volumetric_field = ti.Vector.field(n=3, dtype=ti.f32, shape=box_size)
```

### Access elements in a vector field

Accessing a vector field is similar to accessing a multi-dimensional array: You use an index operator `[]` to access an element in the field. The only difference is that, to access a specific component of an element (vector in this case), you need an *extra* index operator `[]`:

- To access the velocity vector at a specific position of the volumetric field above:

  `volumetric_field[i,j,k]`

- To access the `l`-th component of the velocity vector:

  `volumetric_field[i, j, k][l]`

The following code snippet generates and prints a random vector field:

```python
# n: vector dimension; w: width; h: height
n,w,h = 3,128,64
vec_field = ti.Vector.field(n, dtype=ti.f32, shape=(w,h))

@ti.kernel
def fill_vector():
    for i,j in vec_field:
        for k in ti.static(range(n)):
            #ti.static unrolls the inner loops
            vec_field[i,j][k] = ti.random()

fill_vector()
print(vec_field[w-1,h-1][n-1])
```

:::note
To access the `p`-th component of the 0D vector field `x = ti.Vector.field(n=3, dtype=ti.f32, shape=())`:

`x[None][p]` (0 &le; p < n).
:::

## Matrix fields

As the name suggests, matrix fields are the fields whose elements are matrices. In continuum mechanics, at each infinitesimal point in a 3D material exists a strain and stress tensor. The strain and stress tensor is a 3 x 2 matrix. Then, you can use a matrix field to represent such a tensor field.

### Declaration

The following code snippet declares a tensor field:

```python
# Declare a 300x400x500 matrix field, each of its elements being a 3x2 matrix
tensor_field = ti.Matrix.field(n=3, m=2, dtype=ti.f32, shape=(300, 400, 500))
```

### Access elements in a matrix field

Accessing a matrix field is similar to accessing a vector field: You use an index operator `[]` for field indexing and a second `[]` for matrix indexing.

- To retrieve the `i, j` element of the matrix field `tensor_field`:

  `mat = tensor_field[i, j]`

- To retrieve the member on the first row and second column of the element `mat`:

  `mat[0, 1]` or `tensor_field[i, j][0, 1]`

:::note

To access the 0D matrix field `x = ti.Matrix.field(n=3, m=4, dtype=ti.f32, shape=())`:

`x[None][p, q]` (0 &le; p < n, 0 &le; q < m)

:::

### Considerations: Matrix size

Matrix operations are unrolled during compile time. Take a look at the following example:

```python
import taichi as ti
ti.init()

a = ti.Matrix.field(n=2, m=3, dtype=ti.f32, shape=(2, 2))
@ti.kernel
def test():
    for i in ti.grouped(a):
        # a[i] is a 2x3 matrix
        a[i] = [[1, 1, 1], [1, 1, 1]]
        # The assignment is unrolled to the following during compile time:
        # a[i][0, 0] = 1
        # a[i][0, 1] = 1
        # a[i][0, 2] = 1
        # a[i][1, 0] = 1
        # a[i][1, 1] = 1
        # a[i][1, 2] = 1
```

Operating on large matrices (for example `32x128`) can lead to long compilation time and poor performance. For performance reasons, it is recommended that you keep your matrices small:

- `2x1`, `3x3`, and `4x4` matrices work fine.
- `32x6` is a bit too large.

**Workaround:**

When declaring the matrix field, leave large dimensions to the fields, rather than to the matrices. If you have a `3x2` field of `64x32` matrices:

- Not recommended:
  `ti.Matrix.field(64, 32, dtype=ti.f32, shape=(3, 2))`
- Recommended:
  `ti.Matrix.field(3, 2, dtype=ti.f32, shape=(64, 32))`

## Struct fields

Struct fields are fields that store user-defined structs. Members of a struct element can be:

- Scalars
- Vectors
- Matrices
- Other struct fields.

### Declaration

The following code snippet declares a 1D field of particle information (position, velocity, acceleration, and mass) using `ti.Struct.field()`. Note that:

- Member variables `pos`, `vel`, `acc`, and `mass` are provided in the dictionary format.
- [Compound types](../type_system/type.md#compound-types), such as `ti.types.vector`, `ti.types.matrix`, and `ti.types.struct`, can be used to declare vectors, matrices, or structs as struct members.

```python
# Declare a 1D struct field using the ti.Struct.field() method
particle_field = ti.Struct.field({
    "pos": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(3, ti.f32),
    "acc": ti.types.vector(3, ti.f32),
    "mass": ti.f32,
  }, shape=(n,))
```

Alternatively, besides *directly* using `ti.Struct.field()`, you can first declare a compound type `particle` and then create a field of it:

```python
# Declare a compound type vec3f to represent position, velocity, and acceleration.
vec3f = ti.types.vector(3, ti.f32)
# Declare a struct composed of three vectors and one f32 floating-point number
particle = ti.types.struct(
  pos=vec3f, vel=vec3f, acc=vec3f, mass=ti.f32,
)
# Declare a 1D field of the struct particle using field()
particle_field = particle.field(shape=(n,))
```

### Access elements in a struct field

You can access members of elements in a struct field either one by one or universally:

```python
# Set the position of the first particle in the field to origin [0.0, 0.0, 0.0]
particle_field[0].pos = ti.Vector([0.0, 0.0, 0.0]) # pos is a 3D vector

# Set the second particle's pos[0] in the field to 1.0
particle_field[1].pos[0] = 1.0 # pos[0] is the first member of pos

# Universally set the mass of all particles to 1.0
particle_field.mass.fill(1.0)
```
