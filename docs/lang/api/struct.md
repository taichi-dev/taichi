---
sidebar_position: 3
---

# Structs

Mixed-data-type records can be created in Taichi as custom structs. A struct in Taichi can have two forms. Similar to vectors and matrices, structs have both local variable and global field forms.

## Declaration

### As global vector fields

::: {.function}
ti.struct.field(members, shape = None, offset = None)

parameter members

: (Dict[str, DataType]) name and data type for each struct member. The data type of a member can be either a primitive type (numbers) or a compound type (vectors, matrices, structs).

parameter shape

: (optional, scalar or tuple) shape of the struct field, see
`tensor`{.interpreted-text role="ref"}

parameter offset

: (optional, scalar or tuple) see `offset`{.interpreted-text
role="ref"}

For example, this creates a Struct field of the two float members `a` and `b`: :

    # Python-scope
    x = ti.Struct.field({'a': ti.f32, 'b': fi.f32}, shape=(5, 4))

A struct field with vector, matrix, or struct components can be created with compound types: :

    # Python-scope
    vec3 = ti.types.vector(3, float)
    x = ti.Struct.field({'a': ti.f32, 'b': vec3}, shape=(5, 4))

:::

### As a temporary local variable

A local Struct variable can be created with *either* a dictionary *or* keyword arguments.

::: {.function}
ti.Struct(members, **kwargs)

parameter members

: (Dict) The dictionary containing struct members.

parameter **kwargs

: The keyword arguments to specify struct members.

Lists and nested dictionaries in the member dictionary or keyword arguments will be converted into local `ti.Matrix` and `ti.Struct`, respectively.

For example, this creates a struct with a float member `a` and vector member `b`:

    # Taichi-scope
    x = ti.Struct({'a': 1.0, 'b': [1.0, 1.0, 1.0]})
    # or
    x = ti.Struct(a=1.0, b=ti.Vector([1.0, 1.0, 1.0]))

:::

## Accessing components

### As global struct fields

Global struct field members are accessed as object attributes. For example, this extracts the member `a` of struct `x[6, 3]`: :

    a = x[6, 3].a

    # or
    s = x[6, 3]
    a = s.a

In contrast to vector and matrix fields, struct members in a global struct field can be accessed in both attribute-first and index-first manners:

    a = x[6, 3].a
    # is equivalent to
    a = x.a[6, 3]

This allows for all field elements for a given struct field member to be extracted as a whole as a scalar, vector, or matrix field by accessing the member attributes of *the global struct field`:

    # Python-scope
    vec3 = ti.types.vector(3, float)
    x = ti.Struct.field({'a': ti.f32, 'b': vec3}, shape=(5, 4))

    x.a.fill(1.0) # x.a is equivalent to ti.field(ti.f32, shape=(5, 4))
    x.b.fill(2.0) # x.b is equivalent to ti.Vector.field(3, ti.f32, shape=(5, 4))
    a = x.a.to_numpy()
    b = x.b.to_numpy()

### As a temporary local variable

Members of a local struct can be accessed using both attributes (object-like) or keys (dict-like):

    x = ti.Struct(a=1.0, b=ti.Vector([1.0, 1.0, 1.0]))
    a = x['a'] # a = 1.0
    x.b = ti.Vector([1.0, 2.0, 3.0])

## Element-wise operations (WIP)

TODO: add element wise operations docs
