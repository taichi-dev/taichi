# taichi.lang package

## Submodules

## taichi.lang.ast_checker module


### class taichi.lang.ast_checker.KernelSimplicityASTChecker(func)
Bases: `ast.NodeVisitor`


#### generic_visit(node)
Called if no explicit visitor function exists for a node.

## taichi.lang.ast_resolver module

Provides helpers to resolve AST nodes.


### class taichi.lang.ast_resolver.ASTResolver()
Bases: `object`

Provides helper methods to resolve AST nodes.


#### static resolve_to(node, wanted, scope)
Check if symbol `node` resolves to `wanted` object.

This is only intended to check if a given AST node resolves to a symbol
under some namespaces, e.g. the `a.b.c.foo` pattern, but not meant for
more complicated expressions like `(a + b).foo`.


* **Parameters**

    
    * **node** (*Union**[**ast.Attribute**, **ast.Name**]*) – an AST node to be resolved.


    * **wanted** (*Any*) – The expected python object.


    * **scope** (*Dict**[**str**, **Any**]*) – Maps from symbol names to objects, for
    example, globals()



* **Returns**

    The checked result.



* **Return type**

    bool


## taichi.lang.common_ops module


### class taichi.lang.common_ops.TaichiOperations()
Bases: `object`

The base class of taichi operations of expressions. Subclasses: `Expr`, `Matrix`


#### assign(other)
Assign the expression of the given operand to self.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The expression after assigning.



* **Return type**

    `Expr`



#### atomic_add(other)
Return the new expression of computing atomic add between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of atomic add.



* **Return type**

    `Expr`



#### atomic_and(other)
Return the new expression of computing atomic and between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of atomic and.



* **Return type**

    `Expr`



#### atomic_or(other)
Return the new expression of computing atomic or between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of atomic or.



* **Return type**

    `Expr`



#### atomic_sub(other)
Return the new expression of computing atomic sub between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of atomic sub.



* **Return type**

    `Expr`



#### atomic_xor(other)
Return the new expression of computing atomic xor between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of atomic xor.



* **Return type**

    `Expr`



#### augassign(x, op)
Generate the computing expression between self and the given operand of given operator and assigned to self.


* **Parameters**

    
    * **x** (*Any*) – Given operand.


    * **op** (*str*) – The name of operator.



#### logical_and(other)
Return the new expression of computing logical and between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of logical and.



* **Return type**

    `Expr`



#### logical_or(other)
Return the new expression of computing logical or between self and a given operand.


* **Parameters**

    **other** (*Any*) – Given operand.



* **Returns**

    The computing expression of logical or.



* **Return type**

    `Expr`


## taichi.lang.exception module


### exception taichi.lang.exception.InvalidOperationError()
Bases: `Exception`


### exception taichi.lang.exception.TaichiSyntaxError(\*args)
Bases: `Exception`

## taichi.lang.expr module


### class taichi.lang.expr.Expr(\*args, tb=None)
Bases: `taichi.lang.common_ops.TaichiOperations`

A Python-side Expr wrapper, whose member variable ptr is an instance of C++ Expr class. A C++ Expr object contains member variable expr which holds an instance of C++ Expression class.

## taichi.lang.impl module


### taichi.lang.impl.field(dtype, shape=None, name='', offset=None, needs_grad=False)
Defines a Taichi field

A Taichi field can be viewed as an abstract N-dimensional array, hiding away
the complexity of how its underlying `SNode` are
actually defined. The data in a Taichi field can be directly accessed by
a Taichi `kernel()`.

See also [https://docs.taichi.graphics/docs/lang/articles/basic/field](https://docs.taichi.graphics/docs/lang/articles/basic/field)


* **Parameters**

    
    * **dtype** (*DataType*) – data type of the field.


    * **shape** (*Union**[**int**, **tuple**[**int**]**]**, **optional*) – shape of the field


    * **name** (*str**, **optional*) – name of the field


    * **offset** (*Union**[**int**, **tuple**[**int**]**]**, **optional*) – offset of the field domain


    * **needs_grad** (*bool**, **optional*) – whether this field participates in autodiff
    and thus needs an adjoint field to store the gradients.


### Example

The code below shows how a Taichi field can be declared and defined:

```
>>> x1 = ti.field(ti.f32, shape=(16, 8))
>>>
>>> # Equivalently
>>> x2 = ti.field(ti.f32)
>>> ti.root.dense(ti.ij, shape=(16, 8)).place(x2)
```


### taichi.lang.impl.grouped(x)
Groups a list of independent loop indices into a `Vector()`.


* **Parameters**

    **x** (*Any*) – does the grouping only if x is a `ndrange`.


Example:

```
>>> for I in ti.grouped(ti.ndrange(8, 16)):
>>>     print(I[0] + I[1])
```


### taichi.lang.impl.insert_expr_stmt_if_ti_func(func, \*args, \*\*kwargs)
This method is used only for real functions. It inserts a
FrontendExprStmt to the C++ AST to hold the function call if func is a
Taichi function.


* **Parameters**

    
    * **func** – The function to be called.


    * **args** – The arguments of the function call.


    * **kwargs** – The keyword arguments of the function call.



* **Returns**

    The return value of the function call if it’s a non-Taichi function.
    Returns None if it’s a Taichi function.



### taichi.lang.impl.ndarray(dtype, shape)
Defines a Taichi ndarray with scalar elements.


* **Parameters**

    
    * **dtype** (*DataType*) – Data type of the ndarray.


    * **shape** (*Union**[**int**, **tuple**[**int**]**]*) – Shape of the ndarray.


### Example

The code below shows how a Taichi ndarray with scalar elements can be declared and defined:

```
>>> x = ti.ndarray(ti.f32, shape=(16, 8))
```


### taichi.lang.impl.one(x)
Fill the input field with one.


* **Parameters**

    **x** (*DataType*) – The input field to fill.



* **Returns**

    The output field, which keeps the shape but filled with one.



* **Return type**

    DataType



### taichi.lang.impl.root( = ti.root)
Root of the declared Taichi :func:

```
`
```

~taichi.lang.impl.field\`s.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/layout](https://docs.taichi.graphics/docs/lang/articles/advanced/layout)

Example:

```
>>> x = ti.field(ti.f32)
>>> ti.root.pointer(ti.ij, 4).dense(ti.ij, 8).place(x)
```


### taichi.lang.impl.static(x, \*xs)
Evaluates a Taichi-scope expression at compile time.

static() is what enables the so-called metaprogramming in Taichi. It is
in many ways similar to `constexpr` in C++11.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/meta](https://docs.taichi.graphics/docs/lang/articles/advanced/meta).


* **Parameters**

    
    * **x** (*Any*) – an expression to be evaluated


    * **\*xs** (*Any*) – for Python-ish swapping assignment


### Example

The most common usage of static() is for compile-time evaluation:

```
>>> @ti.kernel
>>> def run():
>>>     if ti.static(FOO):
>>>         do_a()
>>>     else:
>>>         do_b()
```

Depending on the value of `FOO`, `run()` will be directly compiled
into either `do_a()` or `do_b()`. Thus there won’t be a runtime
condition check.

Another common usage is for compile-time loop unrolling:

```
>>> @ti.kernel
>>> def run():
>>>     for i in ti.static(range(3)):
>>>         print(i)
>>>
>>> # The above is equivalent to:
>>> @ti.kernel
>>> def run():
>>>     print(0)
>>>     print(1)
>>>     print(2)
```


### taichi.lang.impl.zero(x)
Fill the input field with zero.


* **Parameters**

    **x** (*DataType*) – The input field to fill.



* **Returns**

    The output field, which keeps the shape but filled with zero.



* **Return type**

    DataType


## taichi.lang.kernel_arguments module


### class taichi.lang.kernel_arguments.ArgAnyArray(element_shape=(), layout=Layout.AOS)
Bases: `object`

Type annotation for arbitrary arrays, including external arrays and Taichi ndarrays.

For external arrays, we can treat it as a Taichi field with Vector or Matrix elements by specifying element shape and layout.


* **Parameters**

    
    * **element_shape** (*Tuple**[**Int**]**, **optional*) – () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.


    * **layout** (*Layout**, **optional*) – Memory layout, AOS by default.



### class taichi.lang.kernel_arguments.ArgExtArray(dim=1)
Bases: `object`

Type annotation for external arrays.

External array is formally defined as the data from other Python frameworks.
For now, Taichi supports numpy and pytorch.


* **Parameters**

    **dim** (*int**, **optional*) – must be 1.



### class taichi.lang.kernel_arguments.Template(tensor=None, dim=None)
Bases: `object`

Type annotation for template kernel parameter.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/meta](https://docs.taichi.graphics/docs/lang/articles/advanced/meta).


* **Parameters**

    
    * **tensor** (*Any*) – unused


    * **dim** (*Any*) – unused



### taichi.lang.kernel_arguments.any_arr()
Alias for `ArgAnyArray`.

Example:

```
>>> @ti.kernel
>>> def to_numpy(x: ti.any_arr(), y: ti.any_arr()):
>>>     for i in range(n):
>>>         x[i] = y[i]
>>>
>>> y = ti.ndarray(ti.f64, shape=n)
>>> ... # calculate y
>>> x = numpy.zeros(n)
>>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
```


### taichi.lang.kernel_arguments.ext_arr()
Alias for `ArgExtArray`.

Example:

```
>>> @ti.kernel
>>> def to_numpy(arr: ti.ext_arr()):
>>>     for i in x:
>>>         arr[i] = x[i]
>>>
>>> arr = numpy.zeros(...)
>>> to_numpy(arr)  # `arr` will be filled with `x`'s data.
```


### taichi.lang.kernel_arguments.template()
Alias for `Template`.

## taichi.lang.kernel_impl module


### exception taichi.lang.kernel_impl.KernelArgError(pos, needed, provided)
Bases: `Exception`


### exception taichi.lang.kernel_impl.KernelDefError(msg)
Bases: `Exception`


### taichi.lang.kernel_impl.data_oriented(cls)
Marks a class as Taichi compatible.

To allow for modularized code, Taichi provides this decorator so that
Taichi kernels can be defined inside a class.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/odop](https://docs.taichi.graphics/docs/lang/articles/advanced/odop)

Example:

```
>>> @ti.data_oriented
>>> class TiArray:
>>>     def __init__(self, n):
>>>         self.x = ti.field(ti.f32, shape=n)
>>>
>>>     @ti.kernel
>>>     def inc(self):
>>>         for i in x:
>>>             x[i] += 1
>>>
>>> a = TiArray(42)
>>> a.inc()
```


* **Parameters**

    **cls** (*Class*) – the class to be decorated



* **Returns**

    The decorated class.



### taichi.lang.kernel_impl.func(fn)
Marks a function as callable in Taichi-scope.

This decorator transforms a Python function into a Taichi one. Taichi
will JIT compile it into native instructions.


* **Parameters**

    **fn** (*Callable*) – The Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable


Example:

```
>>> @ti.func
>>> def foo(x):
>>>     return x + 2
>>>
>>> @ti.kernel
>>> def run():
>>>     print(foo(40))  # 42
```


### taichi.lang.kernel_impl.kernel(fn)
Marks a function as a Taichi kernel.

A Taichi kernel is a function written in Python, and gets JIT compiled by
Taichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
The top-level `for` loops are automatically parallelized, and distributed
to either a CPU thread pool or massively parallel GPUs.

Kernel’s gradient kernel would be generated automatically by the AutoDiff system.

See also [https://docs.taichi.graphics/docs/lang/articles/basic/syntax#kernels](https://docs.taichi.graphics/docs/lang/articles/basic/syntax#kernels).


* **Parameters**

    **fn** (*Callable*) – the Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable


Example:

```
>>> x = ti.field(ti.i32, shape=(4, 8))
>>>
>>> @ti.kernel
>>> def run():
>>>     # Assigns all the elements of `x` in parallel.
>>>     for i in x:
>>>         x[i] = i
```


### taichi.lang.kernel_impl.pyfunc(fn)
Marks a function as callable in both Taichi and Python scopes.

When called inside the Taichi scope, Taichi will JIT compile it into
native instructions. Otherwise it will be invoked directly as a
Python function.

See also `func()`.


* **Parameters**

    **fn** (*Callable*) – The Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable


## taichi.lang.linalg module


### taichi.lang.linalg.eig2x2(A, dt)
Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).


* **Parameters**

    
    * **A** (*ti.Matrix**(**2**, **2**)*) – input 2x2 matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
    eigenvectors: (ti.Matrix(4, 2)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of 2 entries, each of which is represented by two numbers for its real part and imaginary part.



* **Return type**

    eigenvalues (ti.Matrix(2, 2))



### taichi.lang.linalg.polar_decompose(A, dt)
Perform polar decomposition (A=UP) for arbitrary size matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Polar_decomposition](https://en.wikipedia.org/wiki/Polar_decomposition).
2D implementation refers to `taichi.lang.linalg.polar_decompose2d()`.
3D implementation refers to `taichi.lang.linalg.polar_decompose3d()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – input nxn matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed nxn matrices U and P.



### taichi.lang.linalg.polar_decompose2d(A, dt)
Perform polar decomposition (A=UP) for 2x2 matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Polar_decomposition](https://en.wikipedia.org/wiki/Polar_decomposition).


* **Parameters**

    
    * **A** (*ti.Matrix**(**2**, **2**)*) – input 2x2 matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed 2x2 matrices U and P.



### taichi.lang.linalg.polar_decompose3d(A, dt)
Perform polar decomposition (A=UP) for 3x3 matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Polar_decomposition](https://en.wikipedia.org/wiki/Polar_decomposition).


* **Parameters**

    
    * **A** (*ti.Matrix**(**3**, **3**)*) – input 3x3 matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed 3x3 matrices U and P.



### taichi.lang.linalg.svd(A, dt)
Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Singular_value_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).
2D implementation refers to `taichi.lang.linalg.svd2d()`.
3D implementation refers to `taichi.lang.linalg.svd3d()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – input nxn matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed nxn matrices U, ‘S’ and V.



### taichi.lang.linalg.svd2d(A, dt)
Perform singular value decomposition (A=USV^T) for 2x2 matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Singular_value_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).


* **Parameters**

    
    * **A** (*ti.Matrix**(**2**, **2**)*) – input 2x2 matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed 2x2 matrices U, ‘S’ and V.



### taichi.lang.linalg.svd3d(A, dt, iters=None)
Perform singular value decomposition (A=USV^T) for 3x3 matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Singular_value_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).


* **Parameters**

    
    * **A** (*ti.Matrix**(**3**, **3**)*) – input 3x3 matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.


    * **iters** (*int*) – iteration number to control algorithm precision.



* **Returns**

    Decomposed 3x3 matrices U, ‘S’ and V.



### taichi.lang.linalg.sym_eig2x2(A, dt)
Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real symmetric matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).


* **Parameters**

    
    * **A** (*ti.Matrix**(**2**, **2**)*) – input 2x2 symmetric matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    The eigenvalues. Each entry store one eigen value.
    eigenvectors (ti.Matrix(2, 2)): The eigenvectors. Each column stores one eigenvector.



* **Return type**

    eigenvalues (ti.Vector(2))


## taichi.lang.matrix module


### class taichi.lang.matrix.Matrix(n=1, m=1, dt=None, shape=None, offset=None, empty=False, layout=Layout.AOS, needs_grad=False, keep_raw=False, disable_local_tensor=False, rows=None, cols=None)
Bases: `taichi.lang.common_ops.TaichiOperations`

The matrix class.


* **Parameters**

    
    * **n** (*int*) – the first dimension of a matrix.


    * **m** (*int*) – the second dimension of a matrix.


    * **dt** (*DataType*) – the elmement data type.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – the shape of a matrix field.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.


    * **empty** (*Bool**, **deprecated*) – True if the matrix is empty, False otherwise.


    * **layout** (*Layout**, **optional*) – The filed layout (Layout.AOS or Layout.SOA).


    * **needs_grad** (*Bool**, **optional*) – True if used in auto diff, False otherwise.


    * **keep_raw** (*Bool**, **optional*) – Keep the contents in n as is.


    * **rows** (*List**, **deprecated*) – construct matrix rows.


    * **cols** (*List**, **deprecated*) – construct matrix columns.



#### all()
Test whether all element not equal zero.


* **Returns**

    True if all elements are not equal zero, False otherwise.



* **Return type**

    bool



#### any()
Test whether any element not equal zero.


* **Returns**

    True if any element is not equal zero, False otherwise.



* **Return type**

    bool



#### cast(dtype)
Cast the matrix element data type.


* **Parameters**

    **dtype** (*DataType*) – the data type of the casted matrix element.



* **Returns**

    A new matrix with each element’s type is dtype.



#### static cols(cols)
Construct a Matrix instance by concactinating Vectors/lists column by column.


* **Parameters**

    **cols** (*List*) – A list of Vector (1-D Matrix) or a list of list.



* **Returns**

    A `Matrix` instance filled with the Vectors/lists column by column.



* **Return type**

    `Matrix`



#### cross(other)
Perform the cross product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the cross product.



* **Returns**

    The cross product result (1-D Matrix) of the two Vectors.



* **Return type**

    `Matrix`



#### determinant()
Get the determinant of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The determinant of a matrix.



* **Raises**

    **Exception** – Determinants of matrices with sizes >= 5 are not supported.



#### static diag(dim, val)
Construct a diagonal square matrix.


* **Parameters**

    
    * **dim** (*int*) – the dimension of a square matrix.


    * **val** (*TypeVar*) – the diagonal elment value.



* **Returns**

    The constructed diagonal square matrix.



#### dot(other)
Perform the dot product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the dot product.



* **Returns**

    The dot product result (scalar) of the two Vectors.



* **Return type**

    DataType



#### classmethod empty(n, m)
Clear the matrix and fill None.


* **Parameters**

    
    * **n** (*int*) – The number of the row of the matrix.


    * **m** (*int*) – The number of the column of the matrix.



* **Returns**

    A `Matrix` instance filled with None.



* **Return type**

    `Matrix`



#### classmethod field(n, m, dtype, shape=None, name='', offset=None, needs_grad=False, layout=Layout.AOS)
Construct a data container to hold all elements of the Matrix.


* **Parameters**

    
    * **n** (*int*) – The desired number of rows of the Matrix.


    * **m** (*int*) – The desired number of columns of the Matrix.


    * **dtype** (*DataType**, **optional*) – The desired data type of the Matrix.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – The desired shape of the Matrix.


    * **name** (*string**, **optional*) – The custom name of the field.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.


    * **needs_grad** (*bool**, **optional*) – Whether the Matrix need gradients.


    * **layout** (*Layout**, **optional*) – The field layout, i.e., Array Of Structure (AOS) or Structure Of Array (SOA).



* **Returns**

    A `Matrix` instance serves as the data container.



* **Return type**

    `Matrix`



#### fill(val)
Fills the matrix with a specific value in Taichi scope.


* **Parameters**

    **val** (*Union**[**int**, **float**]*) – Value to fill.



#### static identity(dt, n)
Construct an identity Matrix with shape (n, n).


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The number of rows/columns.



* **Returns**

    A n x n identity `Matrix` instance.



* **Return type**

    `Matrix`



#### inverse()
The inverse of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The inverse of a matrix.



* **Raises**

    **Exception** – Inversions of matrices with sizes >= 5 are not supported.



#### inversed()
The inverse of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The inverse of a matrix.



* **Raises**

    **Exception** – Inversions of matrices with sizes >= 5 are not supported.



#### max()
Return the maximum element value.


#### min()
Return the minumum element value.


#### norm(eps=0)
Return the square root of the sum of the absolute squares of its elements.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.


Examples:

```
a = ti.Vector([3, 4])
a.norm() # sqrt(3*3 + 4*4 + 0) = 5
# `a.norm(eps)` is equivalent to `ti.sqrt(a.dot(a) + eps).`
```


* **Returns**

    The square root of the sum of the absolute squares of its elements.



#### norm_inv(eps=0)
Return the inverse of the matrix/vector norm. For norm: please see `norm()`.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.



* **Returns**

    The inverse of the matrix/vector norm.



#### norm_sqr()
Return the sum of the absolute squares of its elements.


#### normalized(eps=0)
Normalize a vector.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.


Examples:

```
a = ti.Vector([3, 4])
a.normalized() # [3 / 5, 4 / 5]
# `a.normalized()` is equivalent to `a / a.norm()`.
```

**NOTE**: Only vector normalization is supported.


#### static one(dt, n, m=None)
Construct a Matrix filled with ones.


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The first dimension (row) of the matrix.


    * **m** (*int**, **optional*) – The second dimension (column) of the matrix.



* **Returns**

    A `Matrix` instance filled with ones.



* **Return type**

    `Matrix`



#### outer_product(other)
Perform the outer product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the outer product.



* **Returns**

    The outer product result (Matrix) of the two Vectors.



* **Return type**

    `Matrix`



#### static rows(rows)
Construct a Matrix instance by concactinating Vectors/lists row by row.


* **Parameters**

    **rows** (*List*) – A list of Vector (1-D Matrix) or a list of list.



* **Returns**

    A `Matrix` instance filled with the Vectors/lists row by row.



* **Return type**

    `Matrix`



#### sum()
Return the sum of all elements.


#### to_numpy(keep_dims=False)
Converts the Matrix to a numpy array.


* **Parameters**

    **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### trace()
The sum of a matrix diagonal elements.


* **Returns**

    The sum of a matrix diagonal elements.



#### transpose()
Get the transpose of a matrix.


* **Returns**

    Get the transpose of a matrix.



#### static unit(n, i, dt=None)
Construct an unit Vector (1-D matrix) i.e., a vector with only one entry filled with one and all other entries zeros.


* **Parameters**

    
    * **n** (*int*) – The length of the vector.


    * **i** (*int*) – The index of the entry that will be filled with one.


    * **dt** (*DataType**, **optional*) – The desired data type.



* **Returns**

    An 1-D unit `Matrix` instance.



* **Return type**

    `Matrix`



#### classmethod var(n, m, dt, \*args, \*\*kwargs)

#### property w()
Get the fourth element of a matrix.


#### classmethod with_entries(n, m, entries)
Construct a Matrix instance by giving all entries.


* **Parameters**

    
    * **n** (*int*) – Number of rows of the matrix.


    * **m** (*int*) – Number of columns of the matrix.


    * **entries** (*List**[**Any**]*) – Given entries.



* **Returns**

    A `Matrix` instance filled with given entries.



* **Return type**

    Matrix



#### property x()
Get the first element of a matrix.


#### property y()
Get the second element of a matrix.


#### property z()
Get the third element of a matrix.


#### static zero(dt, n, m=None)
Construct a Matrix filled with zeros.


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The first dimension (row) of the matrix.


    * **m** (*int**, **optional*) – The second dimension (column) of the matrix.



* **Returns**

    A `Matrix` instance filled with zeros.



* **Return type**

    `Matrix`



### class taichi.lang.matrix.MatrixField(vars, n, m)
Bases: `taichi.lang.field.Field`

Taichi matrix field with SNode implementation.


* **Parameters**

    
    * **vars** (*Expr*) – Field members.


    * **n** (*Int*) – Number of rows.


    * **m** (*Int*) – Number of columns.



#### fill(val)
Fills self with specific values.


* **Parameters**

    **val** (*Union**[**Number**, **List**, **Tuple**, **Matrix**]*) – Values to fill, which should have dimension consistent with self.



#### from_numpy(arr)
Loads all elements from a numpy array.

The shape of the numpy array needs to be the same as self.


* **Parameters**

    **arr** (*numpy.ndarray*) – The source numpy array.



#### get_scalar_field(\*indices)
Creates a ScalarField using a specific field member. Only used for quant.


* **Parameters**

    **indices** (*Tuple**[**Int**]*) – Specified indices of the field member.



* **Returns**

    The result ScalarField.



* **Return type**

    ScalarField



#### to_numpy(keep_dims=False, as_vector=None, dtype=None)
Converts self to a numpy array.


* **Parameters**

    
    * **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    When keep_dims=True, on an n-D matrix field, the numpy array always has n+2 dims, even for 1x1, 1xn, nx1 matrix fields.
    When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.
    For example, a 4x1 or 1x4 matrix field with 5x6x7 elements results in an array of shape 5x6x7x4.


    * **as_vector** (*bool**, **deprecated*) – Whether to make the returned numpy array as a vector, i.e., with shape (n,) rather than (n, 1).
    Note that this argument has been deprecated.
    More discussion about as_vector: [https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858](https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858).


    * **dtype** (*DataType**, **optional*) – The desired data type of returned numpy array.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### to_torch(device=None, keep_dims=False)
Converts self to a torch tensor.


* **Parameters**

    
    * **device** (*torch.device**, **optional*) – The desired device of returned tensor.


    * **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    See `to_numpy()` for more detailed explanation.



* **Returns**

    The result torch tensor.



* **Return type**

    torch.tensor



### taichi.lang.matrix.Vector(n, dt=None, shape=None, offset=None, \*\*kwargs)
Construct a Vector instance i.e. 1-D Matrix.


* **Parameters**

    
    * **n** (*int*) – The desired number of entries of the Vector.


    * **dt** (*DataType**, **optional*) – The desired data type of the Vector.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – The shape of the Vector.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.



* **Returns**

    A Vector instance (1-D `Matrix`).



* **Return type**

    `Matrix`


## taichi.lang.meta module

## taichi.lang.ndrange module

## taichi.lang.ops module


### taichi.lang.ops.abs(a)
The absolute value function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The absolute value of a.



### taichi.lang.ops.acos(a)
The inverses function of cosine.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements in [-1,1].



* **Returns**

    The inverses function of cosine of a.



### taichi.lang.ops.add(a, b)
The add function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    sum of a and b.



### taichi.lang.ops.asin(a)
The inverses function of sine.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements in [-1,1].



* **Returns**

    The inverses function of sine of a.



### taichi.lang.ops.atan2(a, b)
The inverses of the tangent function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The inverses function of tangent of b/a.



### taichi.lang.ops.bit_and(a, b)
Compute bitwise-and


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-and with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.bit_not(a)
The bit not function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Bitwise not of a.



### taichi.lang.ops.bit_or(a, b)
Computes bitwise-or


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-or with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.bit_sar(a, b)
Compute bitwise shift right


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS >> RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.ops.bit_shl(a, b)
Compute bitwise shift left


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS << RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.ops.bit_shr(a, b)
Compute bitwise shift right (in taichi scope)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS >> RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.ops.bit_xor(a, b)
Compute bitwise-xor


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-xor with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.ceil(a)
The ceil function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The least integer greater than or equal to a.



### taichi.lang.ops.cmp_eq(a, b)
Compare two values (equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is equal to RHS, False otherwise.



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.cmp_ge(a, b)
Compare two values (greater than or equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is greater than or equal to RHS, False otherwise



* **Return type**

    bool



### taichi.lang.ops.cmp_gt(a, b)
Compare two values (greater than)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is strictly larger than RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.cmp_le(a, b)
Compare two values (less than or equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is smaller than or equal to RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.cmp_lt(a, b)
Compare two values (less than)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is strictly smaller than RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.cmp_ne(a, b)
Compare two values (not equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is not equal to RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.cos(a)
The cosine function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Cosine of a.



### taichi.lang.ops.exp(a)
The exp function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    e to the a.



### taichi.lang.ops.floor(a)
The floor function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The greatest integer less than or equal to a.



### taichi.lang.ops.floordiv(a, b)
The floor division function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The floor function of a divided by b.



### taichi.lang.ops.get_addr(f, indices)
Query the memory address (on CUDA/x64) of field f at index indices.

Currently, this function can only be called inside a taichi kernel.


* **Parameters**

    
    * **f** (*Union**[**ti.field**, **ti.Vector.field**, **ti.Matrix.field**]*) – Input taichi field for memory address query.


    * **indices** (*Union**[**int**, **ti.Vector**(**)**]*) – The specified field indices of the query.



* **Returns**

    The memory address of f[indices].



* **Return type**

    ti.u64



### taichi.lang.ops.log(a)
The natural logarithm function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements greater than zero.



* **Returns**

    The natural logarithm of a.



### taichi.lang.ops.logical_and(a, b)
Compute bitwise-and


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-and with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.logical_not(a)
The logical not function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    1 iff a=0, otherwise 0.



### taichi.lang.ops.logical_or(a, b)
Computes bitwise-or


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-or with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.ops.max(a, b)
The maxnimum function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The maxnimum of a and b.



### taichi.lang.ops.min(a, b)
The minimum function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The minimum of a and b.



### taichi.lang.ops.mod(a, b)
The remainder function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The remainder of a divided by b.



### taichi.lang.ops.mul(a, b)
The multiply function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a multiplied by b.



### taichi.lang.ops.neg(a)
The negate function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The negative value of a.



### taichi.lang.ops.pow(a, b)
The power function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a to the b.



### taichi.lang.ops.random(dtype=<class 'float'>)
The random function.


* **Parameters**

    **dtype** (*DataType*) – Type of the random variable.



* **Returns**

    A random variable whose type is dtype.



### taichi.lang.ops.raw_div(a, b)
Raw_div function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    If a is a int and b is a int, then return a//b. Else return a/b.



### taichi.lang.ops.raw_mod(a, b)
Raw_mod function. Both a and b can be float.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The remainder of a divided by b.



### taichi.lang.ops.rescale_index(a, b, I)
Rescales the index ‘I’ of field ‘a’ the match the shape of field ‘b’


* **Parameters**

    
    * **a** (*ti.field**(**)**, **ti.Vector.field**, **ti.Matrix.field**(**)*) – input taichi field


    * **b** (*ti.field**(**)**, **ti.Vector.field**, **ti.Matrix.field**(**)*) – output taichi field


    * **I** (*ti.Vector**(**)*) – grouped loop index



* **Returns**

    **Ib** – rescaled grouped loop index



* **Return type**

    ti.Vector()



### taichi.lang.ops.rsqrt(a)
The reciprocal of the square root function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The reciprocal of sqrt(a).



### taichi.lang.ops.sin(a)
The sine function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Sine of a.



### taichi.lang.ops.sqrt(a)
The square root function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not less than zero.



* **Returns**

    x such that x>=0 and x^2=a.



### taichi.lang.ops.sub(a, b)
The sub function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a subtract b.



### taichi.lang.ops.tan(a)
The tangent function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Tangent of a.



### taichi.lang.ops.tanh(a)
The hyperbolic tangent function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    (e\*\*x - e\*\*(-x)) / (e\*\*x + e\*\*(-x)).



### taichi.lang.ops.truediv(a, b)
True division function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The true value of a divided by b.


## taichi.lang.quant_impl module


### class taichi.lang.quant_impl.Quant()
Bases: `object`

Generator of quantized types.

For more details, read [https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf).


#### static fixed(frac, signed=True, range=1.0, compute=None)
Generates a quantized type for fixed-point real numbers.


* **Parameters**

    
    * **frac** (*int*) – Number of bits.


    * **signed** (*bool*) – Signed or unsigned.


    * **range** (*float*) – Range of the number.


    * **compute** (*DataType*) – Type for computation.



* **Returns**

    The specified type.



* **Return type**

    DataType



#### static float(exp, frac, signed=True, compute=None)
Generates a quantized type for floating-point real numbers.


* **Parameters**

    
    * **exp** (*int*) – Number of exponent bits.


    * **frac** (*int*) – Number of fraction bits.


    * **signed** (*bool*) – Signed or unsigned.


    * **compute** (*DataType*) – Type for computation.



* **Returns**

    The specified type.



* **Return type**

    DataType



#### static int(bits, signed=False, compute=None)
Generates a quantized type for integers.


* **Parameters**

    
    * **bits** (*int*) – Number of bits.


    * **signed** (*bool*) – Signed or unsigned.


    * **compute** (*DataType*) – Type for computation.



* **Returns**

    The specified type.



* **Return type**

    DataType



### taichi.lang.quant_impl.quant()
alias of `taichi.lang.quant_impl.Quant`

## taichi.lang.random module


### taichi.lang.random.randn(dt)
Generates a random number from standard normal distribution
using the Box-Muller transform.

## taichi.lang.runtime_ops module

## taichi.lang.shell module

## taichi.lang.snode module


### class taichi.lang.snode.SNode(ptr)
Bases: `object`

A Python-side SNode wrapper.

For more information on Taichi’s SNode system, please check out
these references:


* [https://docs.taichi.graphics/docs/lang/articles/advanced/sparse](https://docs.taichi.graphics/docs/lang/articles/advanced/sparse)


* [https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf)

Arg:

    ptr (pointer): The C++ side SNode pointer.


#### bit_array(axes, dimensions, num_bits)
Adds a bit_array SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.


    * **num_bits** (*int*) – Number of bits to use.



* **Returns**

    The added `SNode` instance.



#### bit_struct(num_bits: int)
Adds a bit_struct SNode as a child component of self.


* **Parameters**

    **num_bits** – Number of bits to use.



* **Returns**

    The added `SNode` instance.



#### bitmasked(axes, dimensions)
Adds a bitmasked SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### deactivate_all()
Recursively deactivate all children components of self.


#### dense(axes, dimensions)
Adds a dense SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### property dtype()
Gets the data type of self.


* **Returns**

    The data type of self.



* **Return type**

    DataType



#### dynamic(axis, dimension, chunk_size=None)
Adds a dynamic SNode as a child component of self.


* **Parameters**

    
    * **axis** (*List**[**Axis**]*) – Axis to activate, must be 1.


    * **dimension** (*int*) – Shape of the axis.


    * **chunk_size** (*int*) – Chunk size.



* **Returns**

    The added `SNode` instance.



#### get_children()
Gets all children components of self.


* **Returns**

    All children components of self.



* **Return type**

    List[SNode]



#### hash(axes, dimensions)
Not supported.


#### property id()
Gets the id of self.


* **Returns**

    The id of self.



* **Return type**

    int



#### lazy_grad()
Automatically place the adjoint fields following the layout of their primal fields.

Users don’t need to specify `needs_grad` when they define scalar/vector/matrix fields (primal fields) using autodiff.
When all the primal fields are defined, using `taichi.root.lazy_grad()` could automatically generate
their corresponding adjoint fields (gradient field).

To know more details about primal, adjoint fields and `lazy_grad()`,
please see Page 4 and Page 13-14 of DiffTaichi Paper: [https://arxiv.org/pdf/1910.00935.pdf](https://arxiv.org/pdf/1910.00935.pdf)


#### loop_range()
Gets the taichi_core.Expr wrapping the taichi_core.GlobalVariableExpression corresponding to self to serve as loop range.


* **Returns**

    See above.



* **Return type**

    taichi_core.Expr



#### property name()
Gets the name of self.


* **Returns**

    The name of self.



* **Return type**

    str



#### property needs_grad()
Checks whether self has a corresponding gradient `SNode`.


* **Returns**

    Whether self has a corresponding gradient `SNode`.



* **Return type**

    bool



#### parent(n=1)
Gets an ancestor of self in the SNode tree.


* **Parameters**

    **n** (*int*) – the number of levels going up from self.



* **Returns**

    The n-th parent of self.



* **Return type**

    Union[None, _Root, SNode]



#### physical_index_position()
Gets mappings from virtual axes to physical axes.


* **Returns**

    Mappings from virtual axes to physical axes.



* **Return type**

    Dict[int, int]



#### place(\*args, offset=None, shared_exponent=False)
Places a list of Taichi fields under the self container.


* **Parameters**

    
    * **\*args** (*List**[**ti.field**]*) – A list of Taichi fields to place.


    * **offset** (*Union**[**Number**, **tuple**[**Number**]**]*) – Offset of the field domain.


    * **shared_exponent** (*bool*) – Only useful for quant types.



* **Returns**

    The self container.



#### pointer(axes, dimensions)
Adds a pointer SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### property shape()
Gets the number of elements from root in each axis of self.


* **Returns**

    The number of elements from root in each axis of self.



* **Return type**

    Tuple[int]



#### property snode()
Gets self.


* **Returns**

    self.



* **Return type**

    SNode


## taichi.lang.tape module

## taichi.lang.transformer module


### class taichi.lang.transformer.ASTTransformerBase(func)
Bases: `ast.NodeTransformer`


### class taichi.lang.transformer.ASTTransformerChecks(func)
Bases: `taichi.lang.transformer.ASTTransformerBase`

## taichi.lang.type_factory_impl module


### class taichi.lang.type_factory_impl.TypeFactory()
Bases: `object`

A Python-side TypeFactory wrapper.


#### custom_float(significand_type, exponent_type=None, compute_type=None, scale=1.0)
Generates a custom float type.


* **Parameters**

    
    * **significand_type** (*DataType*) – Type of significand.


    * **exponent_type** (*DataType*) – Type of exponent.


    * **compute_type** (*DataType*) – Type for computation.


    * **scale** (*float*) – Scaling factor.



* **Returns**

    The specified type.



* **Return type**

    DataType



#### custom_int(bits, signed=True, compute_type=None)
Generates a custom int type.


* **Parameters**

    
    * **bits** (*int*) – Number of bits.


    * **signed** (*bool*) – Signed or unsigned.


    * **compute_type** (*DataType*) – Type for computation.



* **Returns**

    The specified type.



* **Return type**

    DataType


## taichi.lang.util module


### taichi.lang.util.has_pytorch()
Whether has pytorch in the current Python environment.


* **Returns**

    True if has pytorch else False.



* **Return type**

    bool



### taichi.lang.util.to_numpy_type(dt)
Convert taichi data type to its counterpart in numpy.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in numpy.



* **Return type**

    DataType



### taichi.lang.util.to_pytorch_type(dt)
Convert taichi data type to its counterpart in torch.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in torch.



* **Return type**

    DataType



### taichi.lang.util.to_taichi_type(dt)
Convert numpy or torch data type to its counterpart in taichi.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in taichi.



* **Return type**

    DataType


## Module contents


### class taichi.lang.AnyArray(ptr, element_shape, layout)
Bases: `object`

Class for arbitrary arrays in Python AST.


* **Parameters**

    
    * **ptr** (*taichi_core.Expr*) – A taichi_core.Expr wrapping a taichi_core.ExternalTensorExpression.


    * **element_shape** (*Tuple**[**Int**]*) – () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.


    * **layout** (*Layout*) – Memory layout.



#### property shape()
A list containing sizes for each dimension. Note that element shape will be excluded.


* **Returns**

    The result list.



* **Return type**

    List[Int]



### class taichi.lang.AnyArrayAccess(arr, indices_first)
Bases: `object`

Class for first-level access to AnyArray with Vector/Matrix elements in Python AST.


* **Parameters**

    
    * **arr** (*AnyArray*) – See above.


    * **indices_first** (*Tuple**[**Int**]*) – Indices of first-level access.



### class taichi.lang.Expr(\*args, tb=None)
Bases: `taichi.lang.common_ops.TaichiOperations`

A Python-side Expr wrapper, whose member variable ptr is an instance of C++ Expr class. A C++ Expr object contains member variable expr which holds an instance of C++ Expression class.


### class taichi.lang.ExtArray(ptr)
Bases: `object`

Class for external arrays. Constructed by a taichi_core.Expr wrapping a taichi_core.ExternalTensorExpression.


* **Parameters**

    **ptr** (*taichi_core.Expr*) – See above.



#### loop_range()
Gets the corresponding taichi_core.Expr to serve as loop range.

This is not in use now because struct fors on ExtArrays are not supported yet.


* **Returns**

    See above.



* **Return type**

    taichi_core.Expr



#### property shape()
A list containing sizes for each dimension.


* **Returns**

    The result list.



* **Return type**

    List[Int]



### class taichi.lang.Field(vars)
Bases: `object`

Taichi field with SNode implementation.

A field is constructed by a list of field members.
For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
A field member is a Python Expr wrapping a C++ GlobalVariableExpression.
A C++ GlobalVariableExpression wraps the corresponding SNode.


* **Parameters**

    **vars** (*List**[**Expr**]*) – Field members.



#### copy_from(other)
Copies all elements from another field.

The shape of the other field needs to be the same as self.


* **Parameters**

    **other** (*Field*) – The source field.



#### property dtype()
Gets data type of each individual value.


* **Returns**

    Data type of each individual value.



* **Return type**

    DataType



#### fill(val)
Fills self with a specific value.


* **Parameters**

    **val** (*Union**[**int**, **float**]*) – Value to fill.



#### from_numpy(arr)
Loads all elements from a numpy array.

The shape of the numpy array needs to be the same as self.


* **Parameters**

    **arr** (*numpy.ndarray*) – The source numpy array.



#### from_torch(arr)
Loads all elements from a torch tensor.

The shape of the torch tensor needs to be the same as self.


* **Parameters**

    **arr** (*torch.tensor*) – The source torch tensor.



#### get_field_members()
Gets field members.


* **Returns**

    Field members.



* **Return type**

    List[Expr]



#### loop_range()
Gets representative field member for loop range info.


* **Returns**

    Representative (first) field member.



* **Return type**

    taichi_core.Expr



#### property name()
Gets field name.


* **Returns**

    Field name.



* **Return type**

    str



#### parent(n=1)
Gets an ancestor of the representative SNode in the SNode tree.


* **Parameters**

    **n** (*int*) – the number of levels going up from the representative SNode.



* **Returns**

    The n-th parent of the representative SNode.



* **Return type**

    SNode



#### set_grad(grad)
Sets corresponding gradient field.


* **Parameters**

    **grad** (*Field*) – Corresponding gradient field.



#### property shape()
Gets field shape.


* **Returns**

    Field shape.



* **Return type**

    Tuple[Int]



#### property snode()
Gets representative SNode for info purposes.


* **Returns**

    Representative SNode (SNode of first field member).



* **Return type**

    SNode



#### to_numpy(dtype=None)
Converts self to a numpy array.


* **Parameters**

    **dtype** (*DataType**, **optional*) – The desired data type of returned numpy array.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### to_torch(device=None)
Converts self to a torch tensor.


* **Parameters**

    **device** (*torch.device**, **optional*) – The desired device of returned tensor.



* **Returns**

    The result torch tensor.



* **Return type**

    torch.tensor



### class taichi.lang.FieldsBuilder()
Bases: `object`

A builder that constructs a SNodeTree instance.

Example:

```
x = ti.field(ti.i32)
y = ti.field(ti.f32)
fb = ti.FieldsBuilder()
fb.dense(ti.ij, 8).place(x)
fb.pointer(ti.ij, 8).dense(ti.ij, 4).place(y)

# Afer this line, `x` and `y` are placed. No more fields can be placed
# into `fb`.
#
# The tree looks like the following:
# (implicit root)
#  |
#  +-- dense +-- place(x)
#  |
#  +-- pointer +-- dense +-- place(y)
fb.finalize()
```


#### bit_array(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int], num_bits: int)
Same as `taichi.lang.SNode.bit_array()`


#### bit_struct(num_bits: int)
Same as `taichi.lang.SNode.bit_struct()`


#### bitmasked(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.bitmasked()`


#### dense(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.dense()`


#### dynamic(index: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimension: Union[Sequence[int], int], chunk_size: Optional[int] = None)
Same as `taichi.lang.SNode.dynamic()`


#### finalize(raise_warning=True)
Constructs the SNodeTree and finalizes this builder.


* **Parameters**

    **raise_warning** (*bool*) – Raise warning or not.



#### classmethod finalized_roots()
Gets all the roots of the finalized SNodeTree.


* **Returns**

    A list of the roots of the finalized SNodeTree.



#### hash(indices, dimensions)
Same as `taichi.lang.SNode.hash()`


#### lazy_grad()
Same as `taichi.lang.SNode.lazy_grad()`


#### place(\*args: Any, offset: Optional[Union[Sequence[int], int]] = None, shared_exponent: bool = False)
Same as `taichi.lang.SNode.place()`


#### pointer(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.pointer()`


### exception taichi.lang.InvalidOperationError()
Bases: `Exception`


### exception taichi.lang.KernelArgError(pos, needed, provided)
Bases: `Exception`


### exception taichi.lang.KernelDefError(msg)
Bases: `Exception`


### class taichi.lang.Layout(value)
Bases: `enum.Enum`

Layout of a Taichi field or ndarray.

Currently, AOS (array of structures) and SOA (structure of arrays) are supported.


### class taichi.lang.Matrix(n=1, m=1, dt=None, shape=None, offset=None, empty=False, layout=Layout.AOS, needs_grad=False, keep_raw=False, disable_local_tensor=False, rows=None, cols=None)
Bases: `taichi.lang.common_ops.TaichiOperations`

The matrix class.


* **Parameters**

    
    * **n** (*int*) – the first dimension of a matrix.


    * **m** (*int*) – the second dimension of a matrix.


    * **dt** (*DataType*) – the elmement data type.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – the shape of a matrix field.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.


    * **empty** (*Bool**, **deprecated*) – True if the matrix is empty, False otherwise.


    * **layout** (*Layout**, **optional*) – The filed layout (Layout.AOS or Layout.SOA).


    * **needs_grad** (*Bool**, **optional*) – True if used in auto diff, False otherwise.


    * **keep_raw** (*Bool**, **optional*) – Keep the contents in n as is.


    * **rows** (*List**, **deprecated*) – construct matrix rows.


    * **cols** (*List**, **deprecated*) – construct matrix columns.



#### all()
Test whether all element not equal zero.


* **Returns**

    True if all elements are not equal zero, False otherwise.



* **Return type**

    bool



#### any()
Test whether any element not equal zero.


* **Returns**

    True if any element is not equal zero, False otherwise.



* **Return type**

    bool



#### cast(dtype)
Cast the matrix element data type.


* **Parameters**

    **dtype** (*DataType*) – the data type of the casted matrix element.



* **Returns**

    A new matrix with each element’s type is dtype.



#### static cols(cols)
Construct a Matrix instance by concactinating Vectors/lists column by column.


* **Parameters**

    **cols** (*List*) – A list of Vector (1-D Matrix) or a list of list.



* **Returns**

    A `Matrix` instance filled with the Vectors/lists column by column.



* **Return type**

    `Matrix`



#### cross(other)
Perform the cross product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the cross product.



* **Returns**

    The cross product result (1-D Matrix) of the two Vectors.



* **Return type**

    `Matrix`



#### determinant()
Get the determinant of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The determinant of a matrix.



* **Raises**

    **Exception** – Determinants of matrices with sizes >= 5 are not supported.



#### static diag(dim, val)
Construct a diagonal square matrix.


* **Parameters**

    
    * **dim** (*int*) – the dimension of a square matrix.


    * **val** (*TypeVar*) – the diagonal elment value.



* **Returns**

    The constructed diagonal square matrix.



#### dot(other)
Perform the dot product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the dot product.



* **Returns**

    The dot product result (scalar) of the two Vectors.



* **Return type**

    DataType



#### classmethod empty(n, m)
Clear the matrix and fill None.


* **Parameters**

    
    * **n** (*int*) – The number of the row of the matrix.


    * **m** (*int*) – The number of the column of the matrix.



* **Returns**

    A `Matrix` instance filled with None.



* **Return type**

    `Matrix`



#### classmethod field(n, m, dtype, shape=None, name='', offset=None, needs_grad=False, layout=Layout.AOS)
Construct a data container to hold all elements of the Matrix.


* **Parameters**

    
    * **n** (*int*) – The desired number of rows of the Matrix.


    * **m** (*int*) – The desired number of columns of the Matrix.


    * **dtype** (*DataType**, **optional*) – The desired data type of the Matrix.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – The desired shape of the Matrix.


    * **name** (*string**, **optional*) – The custom name of the field.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.


    * **needs_grad** (*bool**, **optional*) – Whether the Matrix need gradients.


    * **layout** (*Layout**, **optional*) – The field layout, i.e., Array Of Structure (AOS) or Structure Of Array (SOA).



* **Returns**

    A `Matrix` instance serves as the data container.



* **Return type**

    `Matrix`



#### fill(val)
Fills the matrix with a specific value in Taichi scope.


* **Parameters**

    **val** (*Union**[**int**, **float**]*) – Value to fill.



#### static identity(dt, n)
Construct an identity Matrix with shape (n, n).


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The number of rows/columns.



* **Returns**

    A n x n identity `Matrix` instance.



* **Return type**

    `Matrix`



#### inverse()
The inverse of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The inverse of a matrix.



* **Raises**

    **Exception** – Inversions of matrices with sizes >= 5 are not supported.



#### inversed()
The inverse of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The inverse of a matrix.



* **Raises**

    **Exception** – Inversions of matrices with sizes >= 5 are not supported.



#### max()
Return the maximum element value.


#### min()
Return the minumum element value.


#### norm(eps=0)
Return the square root of the sum of the absolute squares of its elements.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.


Examples:

```
a = ti.Vector([3, 4])
a.norm() # sqrt(3*3 + 4*4 + 0) = 5
# `a.norm(eps)` is equivalent to `ti.sqrt(a.dot(a) + eps).`
```


* **Returns**

    The square root of the sum of the absolute squares of its elements.



#### norm_inv(eps=0)
Return the inverse of the matrix/vector norm. For norm: please see `norm()`.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.



* **Returns**

    The inverse of the matrix/vector norm.



#### norm_sqr()
Return the sum of the absolute squares of its elements.


#### normalized(eps=0)
Normalize a vector.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.


Examples:

```
a = ti.Vector([3, 4])
a.normalized() # [3 / 5, 4 / 5]
# `a.normalized()` is equivalent to `a / a.norm()`.
```

**NOTE**: Only vector normalization is supported.


#### static one(dt, n, m=None)
Construct a Matrix filled with ones.


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The first dimension (row) of the matrix.


    * **m** (*int**, **optional*) – The second dimension (column) of the matrix.



* **Returns**

    A `Matrix` instance filled with ones.



* **Return type**

    `Matrix`



#### outer_product(other)
Perform the outer product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the outer product.



* **Returns**

    The outer product result (Matrix) of the two Vectors.



* **Return type**

    `Matrix`



#### static rows(rows)
Construct a Matrix instance by concactinating Vectors/lists row by row.


* **Parameters**

    **rows** (*List*) – A list of Vector (1-D Matrix) or a list of list.



* **Returns**

    A `Matrix` instance filled with the Vectors/lists row by row.



* **Return type**

    `Matrix`



#### sum()
Return the sum of all elements.


#### to_numpy(keep_dims=False)
Converts the Matrix to a numpy array.


* **Parameters**

    **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### trace()
The sum of a matrix diagonal elements.


* **Returns**

    The sum of a matrix diagonal elements.



#### transpose()
Get the transpose of a matrix.


* **Returns**

    Get the transpose of a matrix.



#### static unit(n, i, dt=None)
Construct an unit Vector (1-D matrix) i.e., a vector with only one entry filled with one and all other entries zeros.


* **Parameters**

    
    * **n** (*int*) – The length of the vector.


    * **i** (*int*) – The index of the entry that will be filled with one.


    * **dt** (*DataType**, **optional*) – The desired data type.



* **Returns**

    An 1-D unit `Matrix` instance.



* **Return type**

    `Matrix`



#### classmethod var(n, m, dt, \*args, \*\*kwargs)

#### property w()
Get the fourth element of a matrix.


#### classmethod with_entries(n, m, entries)
Construct a Matrix instance by giving all entries.


* **Parameters**

    
    * **n** (*int*) – Number of rows of the matrix.


    * **m** (*int*) – Number of columns of the matrix.


    * **entries** (*List**[**Any**]*) – Given entries.



* **Returns**

    A `Matrix` instance filled with given entries.



* **Return type**

    Matrix



#### property x()
Get the first element of a matrix.


#### property y()
Get the second element of a matrix.


#### property z()
Get the third element of a matrix.


#### static zero(dt, n, m=None)
Construct a Matrix filled with zeros.


* **Parameters**

    
    * **dt** (*DataType*) – The desired data type.


    * **n** (*int*) – The first dimension (row) of the matrix.


    * **m** (*int**, **optional*) – The second dimension (column) of the matrix.



* **Returns**

    A `Matrix` instance filled with zeros.



* **Return type**

    `Matrix`



### class taichi.lang.MatrixField(vars, n, m)
Bases: `taichi.lang.field.Field`

Taichi matrix field with SNode implementation.


* **Parameters**

    
    * **vars** (*Expr*) – Field members.


    * **n** (*Int*) – Number of rows.


    * **m** (*Int*) – Number of columns.



#### fill(val)
Fills self with specific values.


* **Parameters**

    **val** (*Union**[**Number**, **List**, **Tuple**, **Matrix**]*) – Values to fill, which should have dimension consistent with self.



#### from_numpy(arr)
Loads all elements from a numpy array.

The shape of the numpy array needs to be the same as self.


* **Parameters**

    **arr** (*numpy.ndarray*) – The source numpy array.



#### get_scalar_field(\*indices)
Creates a ScalarField using a specific field member. Only used for quant.


* **Parameters**

    **indices** (*Tuple**[**Int**]*) – Specified indices of the field member.



* **Returns**

    The result ScalarField.



* **Return type**

    ScalarField



#### to_numpy(keep_dims=False, as_vector=None, dtype=None)
Converts self to a numpy array.


* **Parameters**

    
    * **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    When keep_dims=True, on an n-D matrix field, the numpy array always has n+2 dims, even for 1x1, 1xn, nx1 matrix fields.
    When keep_dims=False, the resulting numpy array should skip the matrix dims with size 1.
    For example, a 4x1 or 1x4 matrix field with 5x6x7 elements results in an array of shape 5x6x7x4.


    * **as_vector** (*bool**, **deprecated*) – Whether to make the returned numpy array as a vector, i.e., with shape (n,) rather than (n, 1).
    Note that this argument has been deprecated.
    More discussion about as_vector: [https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858](https://github.com/taichi-dev/taichi/pull/1046#issuecomment-633548858).


    * **dtype** (*DataType**, **optional*) – The desired data type of returned numpy array.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### to_torch(device=None, keep_dims=False)
Converts self to a torch tensor.


* **Parameters**

    
    * **device** (*torch.device**, **optional*) – The desired device of returned tensor.


    * **keep_dims** (*bool**, **optional*) – Whether to keep the dimension after conversion.
    See `to_numpy()` for more detailed explanation.



* **Returns**

    The result torch tensor.



* **Return type**

    torch.tensor



### class taichi.lang.SNode(ptr)
Bases: `object`

A Python-side SNode wrapper.

For more information on Taichi’s SNode system, please check out
these references:


* [https://docs.taichi.graphics/docs/lang/articles/advanced/sparse](https://docs.taichi.graphics/docs/lang/articles/advanced/sparse)


* [https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf)

Arg:

    ptr (pointer): The C++ side SNode pointer.


#### bit_array(axes, dimensions, num_bits)
Adds a bit_array SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.


    * **num_bits** (*int*) – Number of bits to use.



* **Returns**

    The added `SNode` instance.



#### bit_struct(num_bits: int)
Adds a bit_struct SNode as a child component of self.


* **Parameters**

    **num_bits** – Number of bits to use.



* **Returns**

    The added `SNode` instance.



#### bitmasked(axes, dimensions)
Adds a bitmasked SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### deactivate_all()
Recursively deactivate all children components of self.


#### dense(axes, dimensions)
Adds a dense SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### property dtype()
Gets the data type of self.


* **Returns**

    The data type of self.



* **Return type**

    DataType



#### dynamic(axis, dimension, chunk_size=None)
Adds a dynamic SNode as a child component of self.


* **Parameters**

    
    * **axis** (*List**[**Axis**]*) – Axis to activate, must be 1.


    * **dimension** (*int*) – Shape of the axis.


    * **chunk_size** (*int*) – Chunk size.



* **Returns**

    The added `SNode` instance.



#### get_children()
Gets all children components of self.


* **Returns**

    All children components of self.



* **Return type**

    List[SNode]



#### hash(axes, dimensions)
Not supported.


#### property id()
Gets the id of self.


* **Returns**

    The id of self.



* **Return type**

    int



#### lazy_grad()
Automatically place the adjoint fields following the layout of their primal fields.

Users don’t need to specify `needs_grad` when they define scalar/vector/matrix fields (primal fields) using autodiff.
When all the primal fields are defined, using `taichi.root.lazy_grad()` could automatically generate
their corresponding adjoint fields (gradient field).

To know more details about primal, adjoint fields and `lazy_grad()`,
please see Page 4 and Page 13-14 of DiffTaichi Paper: [https://arxiv.org/pdf/1910.00935.pdf](https://arxiv.org/pdf/1910.00935.pdf)


#### loop_range()
Gets the taichi_core.Expr wrapping the taichi_core.GlobalVariableExpression corresponding to self to serve as loop range.


* **Returns**

    See above.



* **Return type**

    taichi_core.Expr



#### property name()
Gets the name of self.


* **Returns**

    The name of self.



* **Return type**

    str



#### property needs_grad()
Checks whether self has a corresponding gradient `SNode`.


* **Returns**

    Whether self has a corresponding gradient `SNode`.



* **Return type**

    bool



#### parent(n=1)
Gets an ancestor of self in the SNode tree.


* **Parameters**

    **n** (*int*) – the number of levels going up from self.



* **Returns**

    The n-th parent of self.



* **Return type**

    Union[None, _Root, SNode]



#### physical_index_position()
Gets mappings from virtual axes to physical axes.


* **Returns**

    Mappings from virtual axes to physical axes.



* **Return type**

    Dict[int, int]



#### place(\*args, offset=None, shared_exponent=False)
Places a list of Taichi fields under the self container.


* **Parameters**

    
    * **\*args** (*List**[**ti.field**]*) – A list of Taichi fields to place.


    * **offset** (*Union**[**Number**, **tuple**[**Number**]**]*) – Offset of the field domain.


    * **shared_exponent** (*bool*) – Only useful for quant types.



* **Returns**

    The self container.



#### pointer(axes, dimensions)
Adds a pointer SNode as a child component of self.


* **Parameters**

    
    * **axes** (*List**[**Axis**]*) – Axes to activate.


    * **dimensions** (*Union**[**List**[**int**]**, **int**]*) – Shape of each axis.



* **Returns**

    The added `SNode` instance.



#### property shape()
Gets the number of elements from root in each axis of self.


* **Returns**

    The number of elements from root in each axis of self.



* **Return type**

    Tuple[int]



#### property snode()
Gets self.


* **Returns**

    self.



* **Return type**

    SNode



### class taichi.lang.ScalarField(var)
Bases: `taichi.lang.field.Field`

Taichi scalar field with SNode implementation.


* **Parameters**

    **var** (*Expr*) – Field member.



#### fill(val)
Fills self with a specific value.


* **Parameters**

    **val** (*Union**[**int**, **float**]*) – Value to fill.



#### from_numpy(arr)
Loads all elements from a numpy array.

The shape of the numpy array needs to be the same as self.


* **Parameters**

    **arr** (*numpy.ndarray*) – The source numpy array.



#### to_numpy(dtype=None)
Converts self to a numpy array.


* **Parameters**

    **dtype** (*DataType**, **optional*) – The desired data type of returned numpy array.



* **Returns**

    The result numpy array.



* **Return type**

    numpy.ndarray



#### to_torch(device=None)
Converts self to a torch tensor.


* **Parameters**

    **device** (*torch.device**, **optional*) – The desired device of returned tensor.



* **Returns**

    The result torch tensor.



* **Return type**

    torch.tensor



### class taichi.lang.ScalarNdarray(dtype, shape)
Bases: `taichi.lang.ndarray.Ndarray`

Taichi ndarray with scalar elements implemented with a torch tensor.


* **Parameters**

    
    * **dtype** (*DataType*) – Data type of the ndarray.


    * **shape** (*Union**[**int**, **tuple**[**int**]**]*) – Shape of the ndarray.



#### property shape()
Gets ndarray shape.


* **Returns**

    Ndarray shape.



* **Return type**

    Tuple[Int]



### exception taichi.lang.TaichiSyntaxError(\*args)
Bases: `Exception`


### taichi.lang.Tape(loss, clear_gradients=True)
Return a context manager of `TapeImpl`. The
context manager would catching all of the callings of functions that
decorated by `kernel()` or
`complex_kernel()` under with statement, and calculate
all the partial gradients of a given loss variable by calling all of the
gradient function of the callings caught in reverse order while with
statement ended.

See also `kernel()` and
`complex_kernel()` for gradient functions.


* **Parameters**

    
    * **loss** (`Expr`) – The loss field, which shape should be ().


    * **clear_gradients** (*Bool*) – Before with body start, clear all gradients or not.



* **Returns**

    The context manager.



* **Return type**

    `TapeImpl`


Example:

```
>>> @ti.kernel
>>> def sum(a: ti.float32):
>>>     for I in ti.grouped(x):
>>>         y[None] += x[I] ** a
>>>
>>> with ti.Tape(loss = y):
>>>     sum(2)
```


### taichi.lang.Vector(n, dt=None, shape=None, offset=None, \*\*kwargs)
Construct a Vector instance i.e. 1-D Matrix.


* **Parameters**

    
    * **n** (*int*) – The desired number of entries of the Vector.


    * **dt** (*DataType**, **optional*) – The desired data type of the Vector.


    * **shape** (*Union**[**int**, **tuple of int**]**, **optional*) – The shape of the Vector.


    * **offset** (*Union**[**int**, **tuple of int**]**, **optional*) – The coordinate offset of all elements in a field.



* **Returns**

    A Vector instance (1-D `Matrix`).



* **Return type**

    `Matrix`



### taichi.lang.abs(a)
The absolute value function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The absolute value of a.



### taichi.lang.acos(a)
The inverses function of cosine.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements in [-1,1].



* **Returns**

    The inverses function of cosine of a.



### taichi.lang.add(a, b)
The add function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    sum of a and b.



### taichi.lang.any_arr()
Alias for `ArgAnyArray`.

Example:

```
>>> @ti.kernel
>>> def to_numpy(x: ti.any_arr(), y: ti.any_arr()):
>>>     for i in range(n):
>>>         x[i] = y[i]
>>>
>>> y = ti.ndarray(ti.f64, shape=n)
>>> ... # calculate y
>>> x = numpy.zeros(n)
>>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
```


### taichi.lang.archs_with(archs, \*\*init_kwags)
Run the test on the given archs with the given init args.


* **Parameters**

    
    * **archs** – a list of Taichi archs


    * **init_kwargs** – kwargs passed to ti.init()



### taichi.lang.asin(a)
The inverses function of sine.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements in [-1,1].



* **Returns**

    The inverses function of sine of a.



### taichi.lang.atan2(a, b)
The inverses of the tangent function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The inverses function of tangent of b/a.



### taichi.lang.bit_and(a, b)
Compute bitwise-and


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-and with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.bit_not(a)
The bit not function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Bitwise not of a.



### taichi.lang.bit_or(a, b)
Computes bitwise-or


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-or with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.bit_sar(a, b)
Compute bitwise shift right


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS >> RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.bit_shl(a, b)
Compute bitwise shift left


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS << RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.bit_shr(a, b)
Compute bitwise shift right (in taichi scope)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS >> RHS



* **Return type**

    Union[`Expr`, int]



### taichi.lang.bit_vectorize(arg0: int)

### taichi.lang.bit_xor(a, b)
Compute bitwise-xor


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-xor with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.block_dim(arg0: int)

### taichi.lang.ceil(a)
The ceil function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The least integer greater than or equal to a.



### taichi.lang.clear_all_gradients()
Set all fields’ gradients to 0.


### taichi.lang.clear_kernel_profile_info()
Clear all KernelProfiler records.


### taichi.lang.cmp_eq(a, b)
Compare two values (equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is equal to RHS, False otherwise.



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.cmp_ge(a, b)
Compare two values (greater than or equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is greater than or equal to RHS, False otherwise



* **Return type**

    bool



### taichi.lang.cmp_gt(a, b)
Compare two values (greater than)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is strictly larger than RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.cmp_le(a, b)
Compare two values (less than or equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is smaller than or equal to RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.cmp_lt(a, b)
Compare two values (less than)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is strictly smaller than RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.cmp_ne(a, b)
Compare two values (not equal to)


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    True if LHS is not equal to RHS, False otherwise



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.complex_kernel(func)
A decorator for python function that user can customize the gradient
function by the decorator generated by
`complex_kernel_grad()` for this function, and could be
caught automatically by ti.Tape(). This decorator would not automatically
converted the function to a taichi kernel. Users should call other taichi
kernels if in need to enable automatic parallel computing.


* **Parameters**

    **fn** (*Callable*) – The Python function which needs to be decorated.



* **Returns**

    The decorated function.



* **Return type**

    Callable


Example:

```
>>> @ti.kernel
>>> def multiply(a: ti.float32):
>>>     for I in ti.grouped(x):
>>>         y[I] = x[I] * a
>>>
>>> @ti.kernel
>>> def multiply_grad(a: ti.float32):
>>>     for I in ti.grouped(x):
>>>         x.grad[I] = y.grad[I] / a
>>>
>>> @ti.complex_kernel
>>> def foo(a):
>>>     multiply(a)
>>>
>>> @ti.complex_kernel_grad(foo)
>>> def foo_grad(a):
>>>     multiply_grad(a)
```


### taichi.lang.complex_kernel_grad(primal)
Generate the gradient decorator for a given function decorated by
`complex_kernel()`. See `complex_kernel()`
to get further information and examples.


* **Parameters**

    **primal** (*Callable*) – The primal function for the decorator.



* **Returns**

    The decorator.



* **Return type**

    Callable



### taichi.lang.cos(a)
The cosine function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Cosine of a.



### taichi.lang.cross(self, other)
Perform the cross product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the cross product.



* **Returns**

    The cross product result (1-D Matrix) of the two Vectors.



* **Return type**

    `Matrix`



### taichi.lang.data_oriented(cls)
Marks a class as Taichi compatible.

To allow for modularized code, Taichi provides this decorator so that
Taichi kernels can be defined inside a class.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/odop](https://docs.taichi.graphics/docs/lang/articles/advanced/odop)

Example:

```
>>> @ti.data_oriented
>>> class TiArray:
>>>     def __init__(self, n):
>>>         self.x = ti.field(ti.f32, shape=n)
>>>
>>>     @ti.kernel
>>>     def inc(self):
>>>         for i in x:
>>>             x[i] += 1
>>>
>>> a = TiArray(42)
>>> a.inc()
```


* **Parameters**

    **cls** (*Class*) – the class to be decorated



* **Returns**

    The decorated class.



### taichi.lang.deprecated(old, new, warning_type=<class 'DeprecationWarning'>)
Mark an API as deprecated.


* **Parameters**

    
    * **old** (*str*) – old method.


    * **new** (*str*) – new method.


    * **warning_type** (*builtin warning type*) – type of warning.


Example:

```
>>> @deprecated('ti.sqr(x)', 'x**2')
>>> def sqr(x):
>>>     return x**2
```


* **Returns**

    Decorated fuction with warning message



### taichi.lang.determinant(a)
Get the determinant of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The determinant of a matrix.



* **Raises**

    **Exception** – Determinants of matrices with sizes >= 5 are not supported.



### taichi.lang.dot(self, other)
Perform the dot product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the dot product.



* **Returns**

    The dot product result (scalar) of the two Vectors.



* **Return type**

    DataType



### taichi.lang.eig(A, dt=None)
Compute the eigenvalues and right eigenvectors of a real matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).
2D implementation refers to `taichi.lang.linalg.eig2x2()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – 2D Matrix for which the eigenvalues and right eigenvectors will be computed.


    * **dt** (*DataType*) – The datatype for the eigenvalues and right eigenvectors.



* **Returns**

    The eigenvalues in complex form. Each row stores one eigenvalue. The first number of the eigenvalue represents the real part and the second number represents the imaginary part.
    eigenvectors (ti.Matrix(n\*2, n)): The eigenvectors in complex form. Each column stores one eigenvector. Each eigenvector consists of n entries, each of which is represented by two numbers for its real part and imaginary part.



* **Return type**

    eigenvalues (ti.Matrix(n, 2))



### taichi.lang.exp(a)
The exp function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    e to the a.



### taichi.lang.ext_arr()
Alias for `ArgExtArray`.

Example:

```
>>> @ti.kernel
>>> def to_numpy(arr: ti.ext_arr()):
>>>     for i in x:
>>>         arr[i] = x[i]
>>>
>>> arr = numpy.zeros(...)
>>> to_numpy(arr)  # `arr` will be filled with `x`'s data.
```


### taichi.lang.extension()
alias of `taichi_core.Extension`


### taichi.lang.field(dtype, shape=None, name='', offset=None, needs_grad=False)
Defines a Taichi field

A Taichi field can be viewed as an abstract N-dimensional array, hiding away
the complexity of how its underlying `SNode` are
actually defined. The data in a Taichi field can be directly accessed by
a Taichi `kernel()`.

See also [https://docs.taichi.graphics/docs/lang/articles/basic/field](https://docs.taichi.graphics/docs/lang/articles/basic/field)


* **Parameters**

    
    * **dtype** (*DataType*) – data type of the field.


    * **shape** (*Union**[**int**, **tuple**[**int**]**]**, **optional*) – shape of the field


    * **name** (*str**, **optional*) – name of the field


    * **offset** (*Union**[**int**, **tuple**[**int**]**]**, **optional*) – offset of the field domain


    * **needs_grad** (*bool**, **optional*) – whether this field participates in autodiff
    and thus needs an adjoint field to store the gradients.


### Example

The code below shows how a Taichi field can be declared and defined:

```
>>> x1 = ti.field(ti.f32, shape=(16, 8))
>>>
>>> # Equivalently
>>> x2 = ti.field(ti.f32)
>>> ti.root.dense(ti.ij, shape=(16, 8)).place(x2)
```


### taichi.lang.floor(a)
The floor function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The greatest integer less than or equal to a.



### taichi.lang.floordiv(a, b)
The floor division function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The floor function of a divided by b.



### taichi.lang.func(fn)
Marks a function as callable in Taichi-scope.

This decorator transforms a Python function into a Taichi one. Taichi
will JIT compile it into native instructions.


* **Parameters**

    **fn** (*Callable*) – The Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable


Example:

```
>>> @ti.func
>>> def foo(x):
>>>     return x + 2
>>>
>>> @ti.kernel
>>> def run():
>>>     print(foo(40))  # 42
```


### taichi.lang.get_addr(f, indices)
Query the memory address (on CUDA/x64) of field f at index indices.

Currently, this function can only be called inside a taichi kernel.


* **Parameters**

    
    * **f** (*Union**[**ti.field**, **ti.Vector.field**, **ti.Matrix.field**]*) – Input taichi field for memory address query.


    * **indices** (*Union**[**int**, **ti.Vector**(**)**]*) – The specified field indices of the query.



* **Returns**

    The memory address of f[indices].



* **Return type**

    ti.u64



### taichi.lang.grouped(x)
Groups a list of independent loop indices into a `Vector()`.


* **Parameters**

    **x** (*Any*) – does the grouping only if x is a `ndrange`.


Example:

```
>>> for I in ti.grouped(ti.ndrange(8, 16)):
>>>     print(I[0] + I[1])
```


### taichi.lang.has_pytorch()
Whether has pytorch in the current Python environment.


* **Returns**

    True if has pytorch else False.



* **Return type**

    bool



### taichi.lang.insert_expr_stmt_if_ti_func(func, \*args, \*\*kwargs)
This method is used only for real functions. It inserts a
FrontendExprStmt to the C++ AST to hold the function call if func is a
Taichi function.


* **Parameters**

    
    * **func** – The function to be called.


    * **args** – The arguments of the function call.


    * **kwargs** – The keyword arguments of the function call.



* **Returns**

    The return value of the function call if it’s a non-Taichi function.
    Returns None if it’s a Taichi function.



### taichi.lang.inversed(self)
The inverse of a matrix.

**NOTE**: The matrix dimension should be less than or equal to 4.


* **Returns**

    The inverse of a matrix.



* **Raises**

    **Exception** – Inversions of matrices with sizes >= 5 are not supported.



### taichi.lang.is_arch_supported(arch)
Checks whether an arch is supported on the machine.


* **Parameters**

    **arch** (*taichi_core.Arch*) – Specified arch.



* **Returns**

    Whether arch is supported on the machine.



* **Return type**

    bool



### taichi.lang.is_extension_supported(arch, ext)
Checks whether an extension is supported on an arch.


* **Parameters**

    
    * **arch** (*taichi_core.Arch*) – Specified arch.


    * **ext** (*taichi_core.Extension*) – Specified extension.



* **Returns**

    Whether ext is supported on arch.



* **Return type**

    bool



### taichi.lang.kernel(fn)
Marks a function as a Taichi kernel.

A Taichi kernel is a function written in Python, and gets JIT compiled by
Taichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
The top-level `for` loops are automatically parallelized, and distributed
to either a CPU thread pool or massively parallel GPUs.

Kernel’s gradient kernel would be generated automatically by the AutoDiff system.

See also [https://docs.taichi.graphics/docs/lang/articles/basic/syntax#kernels](https://docs.taichi.graphics/docs/lang/articles/basic/syntax#kernels).


* **Parameters**

    **fn** (*Callable*) – the Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable


Example:

```
>>> x = ti.field(ti.i32, shape=(4, 8))
>>>
>>> @ti.kernel
>>> def run():
>>>     # Assigns all the elements of `x` in parallel.
>>>     for i in x:
>>>         x[i] = i
```


### taichi.lang.kernel_profiler_total_time()
Get elapsed time of all kernels recorded in KernelProfiler.


* **Returns**

    total time in second



* **Return type**

    time (double)



### taichi.lang.log(a)
The natural logarithm function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements greater than zero.



* **Returns**

    The natural logarithm of a.



### taichi.lang.logical_and(a, b)
Compute bitwise-and


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-and with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.logical_not(a)
The logical not function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    1 iff a=0, otherwise 0.



### taichi.lang.logical_or(a, b)
Computes bitwise-or


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – value LHS


    * **b** (Union[`Expr`, `Matrix`]) – value RHS



* **Returns**

    LHS bitwise-or with RHS



* **Return type**

    Union[`Expr`, bool]



### taichi.lang.max(a, b)
The maxnimum function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The maxnimum of a and b.



### taichi.lang.min(a, b)
The minimum function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The minimum of a and b.



### taichi.lang.mod(a, b)
The remainder function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The remainder of a divided by b.



### taichi.lang.mul(a, b)
The multiply function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a multiplied by b.



### taichi.lang.ndarray(dtype, shape)
Defines a Taichi ndarray with scalar elements.


* **Parameters**

    
    * **dtype** (*DataType*) – Data type of the ndarray.


    * **shape** (*Union**[**int**, **tuple**[**int**]**]*) – Shape of the ndarray.


### Example

The code below shows how a Taichi ndarray with scalar elements can be declared and defined:

```
>>> x = ti.ndarray(ti.f32, shape=(16, 8))
```


### taichi.lang.neg(a)
The negate function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The negative value of a.



### taichi.lang.normalized(self, eps=0)
Normalize a vector.


* **Parameters**

    **eps** (*Number*) – a safe-guard value for sqrt, usually 0.


Examples:

```
a = ti.Vector([3, 4])
a.normalized() # [3 / 5, 4 / 5]
# `a.normalized()` is equivalent to `a / a.norm()`.
```

**NOTE**: Only vector normalization is supported.


### taichi.lang.one(x)
Fill the input field with one.


* **Parameters**

    **x** (*DataType*) – The input field to fill.



* **Returns**

    The output field, which keeps the shape but filled with one.



* **Return type**

    DataType



### taichi.lang.outer_product(self, other)
Perform the outer product with the input Vector (1-D Matrix).


* **Parameters**

    **other** (`Matrix`) – The input Vector (1-D Matrix) to perform the outer product.



* **Returns**

    The outer product result (Matrix) of the two Vectors.



* **Return type**

    `Matrix`



### taichi.lang.parallelize(arg0: int)

### taichi.lang.polar_decompose(A, dt=None)
Perform polar decomposition (A=UP) for arbitrary size matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Polar_decomposition](https://en.wikipedia.org/wiki/Polar_decomposition).
This is only a wrapper for `taichi.lang.linalg.polar_decompose()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – input nxn matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed nxn matrices U and P.



### taichi.lang.pow(a, b)
The power function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a to the b.



### taichi.lang.prepare_sandbox()
Returns a temporary directory, which will be automatically deleted on exit.
It may contain the taichi_core shared object or some misc. files.


### taichi.lang.print_kernel_profile_info()
Print the elapsed time(min,max,avg) of Taichi kernels on devices.
To enable this profiler, set kernel_profiler=True in ti.init.

Example:

```
>>> import taichi as ti

>>> ti.init(ti.cpu, kernel_profiler=True)
>>> var = ti.field(ti.f32, shape=1)

>>> @ti.kernel
>>> def compute():
>>>     var[0] = 1.0

>>> compute()
>>> ti.print_kernel_profile_info() #[1]
```

**NOTE**: [1] Currently the result of KernelProfiler could be incorrect on OpenGL
backend due to its lack of support for ti.sync().


### taichi.lang.print_memory_profile_info()
Memory profiling tool for LLVM backends with full sparse support.
This profiler is automatically on.


### taichi.lang.pyfunc(fn)
Marks a function as callable in both Taichi and Python scopes.

When called inside the Taichi scope, Taichi will JIT compile it into
native instructions. Otherwise it will be invoked directly as a
Python function.

See also `func()`.


* **Parameters**

    **fn** (*Callable*) – The Python function to be decorated



* **Returns**

    The decorated function



* **Return type**

    Callable



### taichi.lang.quant()
alias of `taichi.lang.quant_impl.Quant`


### taichi.lang.query_kernel_profile_info(name)
Query kernel elapsed time(min,avg,max) on devices using the kernel name.
To enable this profiler, set kernel_profiler=True in ti.init.


* **Parameters**

    **name** (*str*) – kernel name.



* **Returns**

    struct KernelProfilerQueryResult with member varaibles(counter, min, max, avg)


Example:

```
>>> import taichi as ti

>>> ti.init(ti.cpu, kernel_profiler=True)
>>> n = 1024*1024
>>> var = ti.field(ti.f32, shape=n)

>>> @ti.kernel
>>> def fill():
>>>     for i in range(n):
>>>         var[i] = 0.1

>>> fill()
>>> ti.clear_kernel_profile_info() #[1]
>>> for i in range(100):
>>>     fill()
>>> query_result = ti.query_kernel_profile_info(fill.__name__) #[2]
>>> print("kernel excuted times =",query_result.counter)
>>> print("kernel elapsed time(min_in_ms) =",query_result.min)
>>> print("kernel elapsed time(max_in_ms) =",query_result.max)
>>> print("kernel elapsed time(avg_in_ms) =",query_result.avg)
```

**NOTE**: [1] To get the correct result, query_kernel_profile_info() must be used in conjunction with
clear_kernel_profile_info().

[2] Currently the result of KernelProfiler could be incorrect on OpenGL
backend due to its lack of support for ti.sync().


### taichi.lang.randn(dt=None)
Generates a random number from standard normal distribution.

Implementation refers to `taichi.lang.random.randn()`.


* **Parameters**

    **dt** (*DataType*) – The datatype for the generated random number.



* **Returns**

    The generated random number.



### taichi.lang.raw_div(a, b)
Raw_div function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    If a is a int and b is a int, then return a//b. Else return a/b.



### taichi.lang.raw_mod(a, b)
Raw_mod function. Both a and b can be float.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The remainder of a divided by b.



### taichi.lang.rescale_index(a, b, I)
Rescales the index ‘I’ of field ‘a’ the match the shape of field ‘b’


* **Parameters**

    
    * **a** (*ti.field**(**)**, **ti.Vector.field**, **ti.Matrix.field**(**)*) – input taichi field


    * **b** (*ti.field**(**)**, **ti.Vector.field**, **ti.Matrix.field**(**)*) – output taichi field


    * **I** (*ti.Vector**(**)*) – grouped loop index



* **Returns**

    **Ib** – rescaled grouped loop index



* **Return type**

    ti.Vector()



### taichi.lang.rsqrt(a)
The reciprocal of the square root function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    The reciprocal of sqrt(a).



### taichi.lang.sin(a)
The sine function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Sine of a.



### taichi.lang.sqrt(a)
The square root function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not less than zero.



* **Returns**

    x such that x>=0 and x^2=a.



### taichi.lang.static(x, \*xs)
Evaluates a Taichi-scope expression at compile time.

static() is what enables the so-called metaprogramming in Taichi. It is
in many ways similar to `constexpr` in C++11.

See also [https://docs.taichi.graphics/docs/lang/articles/advanced/meta](https://docs.taichi.graphics/docs/lang/articles/advanced/meta).


* **Parameters**

    
    * **x** (*Any*) – an expression to be evaluated


    * **\*xs** (*Any*) – for Python-ish swapping assignment


### Example

The most common usage of static() is for compile-time evaluation:

```
>>> @ti.kernel
>>> def run():
>>>     if ti.static(FOO):
>>>         do_a()
>>>     else:
>>>         do_b()
```

Depending on the value of `FOO`, `run()` will be directly compiled
into either `do_a()` or `do_b()`. Thus there won’t be a runtime
condition check.

Another common usage is for compile-time loop unrolling:

```
>>> @ti.kernel
>>> def run():
>>>     for i in ti.static(range(3)):
>>>         print(i)
>>>
>>> # The above is equivalent to:
>>> @ti.kernel
>>> def run():
>>>     print(0)
>>>     print(1)
>>>     print(2)
```


### taichi.lang.sub(a, b)
The sub function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    a subtract b.



### taichi.lang.supported_archs()
Gets all supported archs on the machine.


* **Returns**

    All supported archs on the machine.



* **Return type**

    List[taichi_core.Arch]



### taichi.lang.svd(A, dt=None)
Perform singular value decomposition (A=USV^T) for arbitrary size matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Singular_value_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).
This is only a wrappers for `taichi.lang.linalg.svd()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – input nxn matrix A.


    * **dt** (*DataType*) – date type of elements in matrix A, typically accepts ti.f32 or ti.f64.



* **Returns**

    Decomposed nxn matrices U, ‘S’ and V.



### taichi.lang.sym_eig(A, dt=None)
Compute the eigenvalues and right eigenvectors of a real symmetric matrix.

Mathematical concept refers to [https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).
2D implementation refers to `taichi.lang.linalg.sym_eig2x2()`.


* **Parameters**

    
    * **A** (*ti.Matrix**(**n**, **n**)*) – Symmetric Matrix for which the eigenvalues and right eigenvectors will be computed.


    * **dt** (*DataType*) – The datatype for the eigenvalues and right eigenvectors.



* **Returns**

    The eigenvalues. Each entry store one eigen value.
    eigenvectors (ti.Matrix(n, n)): The eigenvectors. Each column stores one eigenvector.



* **Return type**

    eigenvalues (ti.Vector(n))



### taichi.lang.tan(a)
The tangent function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    Tangent of a.



### taichi.lang.tanh(a)
The hyperbolic tangent function.


* **Parameters**

    **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.



* **Returns**

    (e\*\*x - e\*\*(-x)) / (e\*\*x + e\*\*(-x)).



### taichi.lang.template()
Alias for `Template`.


### taichi.lang.to_numpy_type(dt)
Convert taichi data type to its counterpart in numpy.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in numpy.



* **Return type**

    DataType



### taichi.lang.to_pytorch_type(dt)
Convert taichi data type to its counterpart in torch.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in torch.



* **Return type**

    DataType



### taichi.lang.to_taichi_type(dt)
Convert numpy or torch data type to its counterpart in taichi.


* **Parameters**

    **dt** (*DataType*) – The desired data type to convert.



* **Returns**

    The counterpart data type in taichi.



* **Return type**

    DataType



### taichi.lang.tr(self)
The sum of a matrix diagonal elements.


* **Returns**

    The sum of a matrix diagonal elements.



### taichi.lang.truediv(a, b)
True division function.


* **Parameters**

    
    * **a** (Union[`Expr`, `Matrix`]) – A number or a matrix.


    * **b** (Union[`Expr`, `Matrix`]) – A number or a matrix with elements not equal to zero.



* **Returns**

    The true value of a divided by b.



### taichi.lang.vectorize(arg0: int)

### taichi.lang.warning(msg, type=<class 'UserWarning'>, stacklevel=1)
Print warning message


* **Parameters**

    
    * **msg** (*str*) – massage to print.


    * **type** (*builtin warning type*) – type of warning.


    * **stacklevel** (*int*) – warning stack level from the caller.



### taichi.lang.zero(x)
Fill the input field with zero.


* **Parameters**

    **x** (*DataType*) – The input field to fill.



* **Returns**

    The output field, which keeps the shape but filled with zero.



* **Return type**

    DataType
