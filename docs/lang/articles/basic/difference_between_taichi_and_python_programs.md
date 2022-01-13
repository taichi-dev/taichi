---
sidebar_position: 2
---

# Difference between taichi and python programs

Although Taichi uses Python as the frontend, there are some differences between taichi and python programs.
Main differences are:

1. Taichi only supports one return statement outside non-static `if`/`for`/`while` scope in the program while Python supports return statements in other places.
2. Taichi uses lexical scoping (static scoping) while python uses dynamic scoping.
3. Taichi does not support some of Python's language features.

## Return statement

- If a taichi kernel/function does not have a return statement, it must not have return type annotation.
- If a taichi kernel has a return statement, it must have return type annotation.
- If a taichi function has a return statement, return type annotation is recommended, and it will be mandatory in the future.

```python {3, 7, 10, 14}
@ti.kernel
def error_kernel_no_return_annotation():
    return 0  # Error: Have return statement but have no return type annotation

@ti.func
def ok_func_no_return_annotation():
    return 0  # Ok: Return type annotation of function is optional

@ti.kernel
def error_kernel_no_return() -> ti.i31:  # Error: Have return type annotation but have no return statement
    pass

@ti.func
def error_func_no_return() -> ti.i31:  # Error: Have return type annotation but have no return statement
    pass
```

- The return statement can not be in non-static `if`/`for`/`while` scope.

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
- If there are [compile-time evaluations](/lang/articles/advanced/meta#compile-time-evaluations) in the code, make sure there is a return statement in all circumstances.
Otherwise, error occurs when the branch that does not have return statement is chosen.
```python {7-8, 15-16}
@ti.kernel
def return_inside_static_if(a: ti.template()) -> ti.i32:
    if ti.static(a):
        return 1
    return 0

return_inside_static_if(1)  # OK: returns 1
return_inside_static_if(0)  # OK: returns 0

@ti.kernel
def return_inside_static_if_no_return_outside(a: ti.template()) -> ti.i32:
    if ti.static(a):
        return 1

return_inside_static_if_no_return_outside(1)  # OK: returns 1
return_inside_static_if_no_return_outside(0)  # Error: No return statement
```
