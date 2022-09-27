---
sidebar_position: 1
---

# Fields

The term _field_ is borrowed from mathematics and physics. If you already know [scalar field](https://en.wikipedia.org/wiki/Scalar_field) (for example heat field), or vector field (for example [gravitational field](https://en.wikipedia.org/wiki/Gravitational_field)), then it is easy for you to understand fields in Taichi.

Fields in Taichi are the _global_ data containers that can be accessed from both the Python scope and the Taichi scope. Just like an ndarray in NumPy or a tensor in PyTorch, a field in Taichi is defined as a multi-dimensional array of elements, and elements in a field can be a scalar, a vector, a matrix, or a struct.

:::note
A 0D (zero-dimensional) field contains *only* one element.
:::

## Scalar fields

Scalar fields refer to the fields that store scalars and are the most basic fields.

- A 0D scalar field is a single scalar.
- A 1D scalar field is a 1D array of scalars.
- A 2D scalar field is a 2D array of scalars, and so on.


### Declaration

The simplest way to declare a scalar field is to call `ti.field(dtype, shape)`, where `dtype` is a primitive data type as explained in [type system](../type_system/type.md) and `shape` is a tuple of integers.

- To declare a 0D scalar field, set its shape to the empty tuple `()`:

  ```python
  # Declare a 0D scalar field whose data type is f32
  f_0d = ti.field(ti.f32, shape=())  # 0D field
  ```

  An illustration of `f_0d`:

  ```
      ┌─────┐
      │     │
      └─────┘
      └─────┘
   f_0d.shape=()
  ```

- To declare a 1D scalar field of length `n`, set its shape to `n` or `(n,)`:

  ```python
  f_1d = ti.field(ti.i32, shape=9)  # 1D field
  ```

  An illustration of `f_1d`:

  ```
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │   │   │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┘
  └───────────────────────────────────┘
          f_1d.shape = (9,)
  ```

    There is no essential difference between a 0D field and a 1D field of length 1, except the indexing rules. You *must* use `None` as the index to access a 0D field and use `0` as the index to access a 1D field of length 1:

    ```python
    # Basially f1 and f2 are interchangeable
    f1 = ti.field(int, shape=())
    f2 = ti.field(int, shape=1)

    f1[None] = 1  # Use None to access a 0D field
    f2[0] = 1  # Use 0 to access a 1D field of length 1
    ```

- To declare a 2D scalar field, set its two dimensions respectively, i.e., the numbers of rows and columns. For example, the following code snippet defines a 2D scalar field of shape (3, 6) (of 3 rows and 6 columns):

  ```python
  f_2d = ti.field(int, shape=(3, 6))  # 2D field
  ```

  An illustration of `f_2d` is shown below:

  ```
                         f_2d.shape[1]
                             (=6)
                   ┌───────────────────────┐

                ┌  ┌───┬───┬───┬───┬───┬───┐  ┐
                │  │   │   │   │   │   │   │  │
                │  ├───┼───┼───┼───┼───┼───┤  │
  f_2d.shape[0] │  │   │   │   │   │   │   │  │
      (=3)      │  ├───┼───┼───┼───┼───┼───┤  │
                │  │   │   │   │   │   │   │  │
                └  └───┴───┴───┴───┴───┴───┘  ┘
  ```

Scalar fields of higher dimensions can be similarily defined.

:::caution WARNING

Taichi only supports fields of dimensions &le; 8.

:::


### Access elements in a scalar field

Once a field is declared, Taichi automatically initializes its elements with the value zero.

To access an element in a scalar field, you need to explicitly specify the element's index.

:::note
When accessing a 0D field `x`, use `x[None] = 0`, *not* `x = 0`.
:::

- To access the element in a 0D field, use the index `None` even though the field has only one element:

  ```python
  f_0d[None] = 10.0
  ```

  The layout of `f_0d` is like:

  ```
      ┌──────┐
      │ 10.0 │
      └──────┘
      └──────┘
    f_0d.shape=()
  ```

- To access an element in a 1D field, use index `i`, which is an integer in the range `[0, f_1d.shape[0])`:

  ```python
  for i in range(9):
      f_1d[i] = i
  ```

  The layout of `f_1d` is like:

  ```
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ```

- To access an element in a 2D field, use index `(i, j)`, which is an integer pair.

  -  `i` in the range `[0, f_2d.shape[0])`
  -  `j` in the range `[0, f_2d.shape[1])`:

  ```python
  for i, j in f_2d:
      f_2d[i, j] = i
  ```

  The layout of `f_2d` is like:

  ```
  ┌───┬───┬───┬───┬───┬───┐
  │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
  ├───┼───┼───┼───┼───┼───┤
  │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
  ├───┼───┼───┼───┼───┼───┤
  │ 2 │ 2 │ 2 │ 2 │ 2 │ 2 │
  └───┴───┴───┴───┴───┴───┘
  ```

- To access an element in an n-dimensional field, use index `(i, j, k, ...)`, which is an n-tuple of integers.

You can use a 2D scalar field to represent a 2D grid of values. The following code snippet creates and displays a 640&times;480 gray scale image of randomly-generated values:

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

:::caution WARNING

Taichi does not support slicing on a Taichi field. Neither of the following usage is correct:

```python
for x in f_2d[0]:  # Error! You tried to access its first row，but it's not supported
    ...
```

```python
f_2d[0][3:] = [4, 5, 6]  # Error! You tried to access a slice of the first row, but it's not supported
```

*Either way, an error message 'Slicing is not supported on ti.field' occurs.*
:::


### Fill a scalar field with a given value

To set all elements in a scalar field to a given value, call `field.fill()`:

```python
x = ti.field(int, shape=(5, 5))
x.fill(1)  # Set all elements in x to 1

@ti.kernel
def test():
    x.fill(-1)  # Set all elements in x to -1
```

### Metadata

Metadata provides the basic information of a scalar field. You can retrieve the data type and shape of a scalar field via its `shape` and `dtype` properties:

```python
f_1d.shape  # (9,)
f_3d.dtype  # f32
```

## Vector fields

As the name suggests, vector fields are the fields whose elements are vectors. What a vector represents depends on the scenario of your program. For example, a vector may stand for the (R, G, B) triple of a pixel, the position of a particle, or the gravitational field in space.

### Declaration

Declaring a vector field where each element is an `N`-dimensional vector is similar to the way you declare a scalar field, except that you need to call the function `ti.Vector.field` instead of `ti.field` and specify `N` as the first positional argument.

For example, the following code snippet declares a 2D field of 2D vectors:

```python
# Declare a 3x3 vector field comprising 2D vectors
f = ti.Vector.field(n=2, dtype=float, shape=(3, 3))
```

The layout of `f` is like:

```
                     f.shape[1]
                       (=3)
               ┌────────────────────┐

            ┌  ┌──────┬──────┬──────┐  ┐
            │  │[*, *]│[*, *]│[*, *]│  │
            │  ├──────┼──────┼──────┤  │
 f.shape[0] │  │[*, *]│[*, *]│[*, *]│  │     [*,  *]
    (=3)    │  ├──────┼──────┼──────┤  │     └─────┘
            │  │[*, *]│[*, *]│[*, *]│  │       n=2
            └  └──────┴──────┴──────┘  ┘
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

  `volumetric_field[i, j, k]`

- To access the `l`-th component of the velocity vector:

  `volumetric_field[i, j, k][l]`

- Alternatively, you can use swizzling with the indices `xyzw` or `rgba` to access the components of a vector, provided that the dimension of the vector is no more than four:

  ```python
  volumetric_field[i, j, k].x = 1  # equivalent to volumetric_field[i, j, k][0] = 1
  volumetric_field[i, j, k].y = 2  # equivalent to volumetric_field[i, j, k][1] = 2
  volumetric_field[i, j, k].z = 3  # equivalent to volumetric_field[i, j, k][2] = 3
  volumetric_field[i, j, k].w = 4  # equivalent to volumetric_field[i, j, k][3] = 4
  volumetric_field[i, j, k].xyz = 1, 2, 3  # assign 1, 2, 3 to the first three components
  volumetric_field[i, j, k].rgb = 1, 2, 3  # equivalent to the above
  ```


The following code snippet generates and prints a random vector field:

```python
# n: vector dimension; w: width; h: height
n, w, h = 3, 128, 64
vec_field = ti.Vector.field(n, dtype=float, shape=(w,h))

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

Operating on larger matrices (for example `32x128`) can lead to longer compilation time and poorer performance. For performance reasons, it is recommended that you keep your matrices small:

- `2x1`, `3x3`, and `4x4` matrices work fine.
- `32x6` is a bit too large.

**Workaround:**

When declaring a matrix field, leave large dimensions to the fields, rather than to the matrices. If you have a `3x2` field of `64x32` matrices:

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
    "pos": ti.math.vec3,
    "vel": ti.math.vec3,
    "acc": ti.math.vec3,
    "mass": float,
  }, shape=(n,))
```

Besides *directly* using `ti.Struct.field()`, you can first declare a compound type `particle` and then create a field of it:

```python
# vec3 is a built-in vector type suppied in the `taichi.math` module
vec3 = ti.math.vec3
# Declare a struct composed of three vectors and one floating-point number
particle = ti.types.struct(
  pos=vec3, vel=vec3, acc=vec3, mass=float,
)
# Declare a 1D field of the struct particle using field()
particle_field = particle.field(shape=(n,))
```

### Access elements in a struct field

You can access a member of an element in a struct field in either of the following ways: index-first or name-first.

+ The index-first approach locates a certain element with its index before specifying the name of the target member:

```python
# Set the position of the first particle in the field to origin [0.0, 0.0, 0.0]
particle_field[0].pos = vec3(0) # pos is a 3D vector
```

The name-first approach, in contrast, first creates the sub-field that gathers all the `mass` members in the struct field and then uses the index to access a specific one:

```python
particle_field.mass[0] = 1.0  # Set the mass of the first particle in the field to 1.0
```

Considering that `paticle_field.mass` is a field consisting of all the `mass` members of the structs in `paticle_field`, you can also call `fill()` to set the members to a specific value all at once:

```python
particle_field.mass.fill(1.0)  # Set all mass of the particles in the struct field to 1.0
```
