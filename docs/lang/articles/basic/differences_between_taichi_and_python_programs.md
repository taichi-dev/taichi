---
sidebar_position: 2
---

# Differences between Taichi and Python programs

Although Taichi uses Python as the frontend, it follows a different set of rules in many aspects, including:

1. [Taichi only supports return statement outside non-static `if`/`for`/`while` scope in the program](#return-statement)
2. [Variables defined inside an `if`/`for`/`while` block cannot be accessed outside the block.](#variable-scoping)
3. [Taichi does not fully support some language features of Python.](#unsupportedpartially-supported-python-language-features)
  - [Set, list, dictionary and operator `in`](#set-list-dictionary-and-operator-in)
  - [Comprehensions](#comprehensions)
  - [Operator `is`](#operator-is)

## Return statement and return type annotation

- If a Taichi kernel/function does not have a return statement, it must not have return type annotation.
- If a Taichi kernel has a return statement, it must have return type annotation.
- If a Taichi function has a return statement, return type annotation is recommended, and it will be mandatory in the future.

```python {3,7,10,14}
@ti.kernel
def error_kernel_no_return_annotation():
    return 0  # Error: Have return statement but have no return type annotation

@ti.kernel
def error_kernel_no_return() -> ti.i32:  # Error: Have return type annotation but have no return statement
    pass

@ti.func
def error_func_no_return() -> ti.i32:  # Error: Have return type annotation but have no return statement
    pass
```

- The return statement can not be in a scope of non-static `if`/`for`/`while`.

```python {4}
@ti.kernel
def error_return_inside_non_static_if(a: ti.i32) -> ti.i32:
    if a:
        return 1  # Error: Return statement inside if scope
```

- The compiler discards code after the first return statement.

```python {4-5}
@ti.kernel
def discarded_after_first_return(a: ti.i32) -> ti.i32:
    return 1
    if a:  # Discarded
        return 1  # Discarded

discarded_after_first_return(0)  # OK: returns 1
```
- If there are [compile-time evaluations](../advanced/meta.md#compile-time-evaluations) in the code, make sure there is a return statement under all circumstances.
Otherwise, error occurs when a branch is chosen which does not have return statement.
```python {7-8,15-16,21,23-24}
@ti.kernel
def return_inside_static_if(a: ti.template()) -> ti.i32:
    if ti.static(a):
        return 1
    return 0

return_inside_static_if(1)  # OK: Returns 1
return_inside_static_if(0)  # OK: Returns 0

@ti.kernel
def return_inside_static_if_no_return_outside(a: ti.template()) -> ti.i32:
    if ti.static(a):
        return 1

return_inside_static_if_no_return_outside(1)  # OK: Returns 1
return_inside_static_if_no_return_outside(0)  # Error: No return statement

@ti.kernel
def ok_return_inside_static_for() -> ti.i32:
    a = 0
    for i in ti.static(range(10)):  # Static for
        a += i
        if ti.static(i == 8):  # Static if
            return a  # OK: Returns 36
```

## Variable scoping

In Python, a variable defined inside an `if`/`for`/`while` block can be accessed outside the block.
**However**, in Taichi, the variables can only be accessed **within the block it is defined**.

```python {5,13,17,22}
@ti.kernel
def error_access_var_outside_for() -> ti.i32:
    for i in range(10):
        a = i
    return a  # Error: variable "a" not found

@ti.kernel
def error_access_var_outside_if(a: ti.i32) -> ti.i32:
    if a:
        b = 1
    else:
        b = 2
    return b  # Error: variable "b" not found

@ti.kernel
def ok_define_var_before_if(a: ti.i32) -> ti.i32:
    b = 0
    if a:
        b = 1
    else:
        b = 2
    return b  # OK: "b" is defined before "if"

ok_define_var_before_if(0)  # Returns 2
```

## Unsupported/partially supported Python language features

### Set, list, dictionary and operator `in`

Currently, Taichi does not support `set`.

List and dictionary before assigning to a variable works as the python list and dictionary.
However, after assigning to a variable, the content of the list and the values (not keys) of the dictionary are converted to Taichi variables.

Taichi does not have a runtime implementation of `in` currently. Therefore, operator `in` and `not in` only works in  [static scope](../advanced/meta.md#static-scope) (inside `ti.static()`).

```python {3,11-12,20}
@ti.kernel
def list_without_assign() -> ti.i32:
    if ti.static(1 in [1, 2]):  # [1, 2]
        return 1
    return 0

list_without_assign()  # Returns 1

@ti.kernel
def list_assigned() -> ti.i32:
    a = [1, 2]  # a: [Variable(1), Variable(2)]
    if ti.static(1 in a):  # 1 is not in [Variable(1), Variable(2)]
        return 1
    return 0

list_assigned()  # Returns 0

@ti.kernel
def error_non_static_in():
    if i in [1, 2]:  # Error: Cannot use `in` outside static scope
        pass
```

### Comprehensions

Taichi partially supports list comprehension and dictionary comprehension,
but does not support set comprehension.

For list comprehensions and dictionary comprehensions, the `if`s and `for`s in them are evaluated at compile time.
The iterators and conditions are implicitly in [static scope](../advanced/meta.md#static-scope).

### Operator `is`

Currently, Taichi does not support operator `is` and `is not`.
