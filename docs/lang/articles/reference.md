---
sidebar_position: 9998
---

# Language Reference

This article describes the syntax and semantics of the Taichi programming
language.

**To users**: If you have gone through user tutorials and still feel uncertain
about your program behavior, then you are in the right place. If you find the
actual behavior different from the one described in this article, feel free to
create an [issue](https://github.com/taichi-dev/taichi/issues/new/choose).
Anything unspecified in this article is subject to change, so you should not
rely on it in your programs.

**To contributors**: This article specifies what the language *should* be. That
is, you should try to match the implementation of the Taichi compiler with this
article. You can clearly determine a certain behavior is *correct*, *buggy*, or
*undefined* from this article.

## Introduction

Taichi is a domain-specific language embedded in Python.
[Kernels and functions](basic/syntax.md) clearly defines the boundary between
the Taichi language and the Python language - code in the Taichi scope is
treated as the former, while code in the Python scope is treated as the latter.
It should be emphasized that this article is about *the Taichi language*.

That said, because Taichi is embedded in Python, the syntax of Taichi is a
subset of that of Python. To make life easier, this article is modeled after
the [Python language reference](https://docs.python.org/3/reference/). The
[notation](https://docs.python.org/3/reference/introduction.html#notation) and
[lexical analysis](https://docs.python.org/3/reference/lexical_analysis.html)
parts exactly follow Python. Please familiarize yourself with them if they seem
new.

## Basic concepts

Before detailing syntax and semantics in the next few chapters, many basic but
important concepts and general evaluation principles specific to Taichi are
listed here.

### Values and types

Like many other programming languages, each expression in Taichi will be
evaluated to a value, and each value has a type. Because Taichi provides easy
interaction with Python and [meta-programming](advanced/meta.md) support, there
are actually two kinds of evaluation: *compile-time evaluation* and *runtime
evaluation*. There are also two kinds of values: *Python values* and *Taichi
values*.

:::note
For readers familiar with programming language terms, such behavior is inspired
by [multi-stage programming](https://en.wikipedia.org/wiki/Multi-stage_programming)
or [partial evaluation](https://en.wikipedia.org/wiki/Partial_evaluation).
:::

A Python value is simply a
[Python object](https://docs.python.org/3/reference/datamodel.html),
which directly comes from the following sources:
- Literals
- Arguments passed via `ti.template()`
- Free variables

Furthermore, as long as all the operands of an operation are Python values,
compile-time evaluation will take place, producing a result Python value. For
meta-programming purposes, Taichi provides an advanced environment for
compile-time evaluation: `ti.static()`, where more operations are supported.

A Python value only exists at compile time. After compile-time evaluation, all
the remaining expressions will be evaluated to Taichi values at runtime.

A Taichi value has a Taichi type, which is one of the following:
- A primitive type, as described in [Type system](basic/type.md)
- A compound type, as described in [Type system](basic/type.md)
- An ndarray type, as introduced in [Run a Taichi Program using Ndarray on
Android](tutorials/ndarray_android.md)
- A sparse matrix builder type, as introduced in [Sparse
Matrix](advanced/sparse_matrix.md)

:::note
An informal quick summary of evaluation rules:
- Python value + Python value = Python value
- Python value + Taichi value = Taichi value
- Taichi value + Taichi value = Taichi value
:::

### Variables and scope

A variable contains a *name*, a *type* and a *value*. In Taichi, a variable can
be defined in the following ways:
- A parameter. The name of the variable is the parameter name. The type of the
variable is the parameter type annotation. The value of the variable is passed
in at runtime.
- An [assignment](#assignment-statements) statement, if the name on the
left-hand side appears for the first time. The name of the variable is the name
on the left-hand side. If there is a type annotation on the left-hand side, the
type of the variable is the type annotation; otherwise, the type of the
variable is inferred from the expression on the right-hand side. The value of
the variable is the evaluation result of the expression on the right-hand side
at runtime.

Taichi is statically-typed. That is, you cannot change the type of a variable
after its definition. However, you can change the value of a variable if there
is another assignment statement after its definition.

Taichi adopts [lexical scope](https://en.wikipedia.org/wiki/Scope_(computer_science)).
Therefore, if a variable is defined in a [block](#compound-statements), it is
invisible outside that block.

## Expressions

The section explains the syntax and semantics of expressions in Taichi.

### Atoms

Atoms are the most basic elements of expressions. The simplest atoms are
identifiers or literals. Forms enclosed in parentheses, brackets or braces
are also categorized syntactically as atoms.

```
atom      ::= identifier | literal | enclosure
enclosure ::= parenth_form | list_display | dict_display
```

#### Identifiers (Names)

Lexical definition of
[identifiers](https://docs.python.org/3/reference/lexical_analysis.html#identifiers)
(also referred to as names) in Taichi follows Python.

There are three cases during evaluation:
- The name is visible and corresponds to a variable defined in Taichi. Then the
evaluation result is the value of the variable at runtime.
- The name is only visible in Python, i.e., the name binding is outside Taichi.
Then compile-time evaluation is triggered, resulting in the Python value bound
to that name.
- The name is invisible. Then a `TaichiNameError` is thrown.

#### Literals

Taichi supports [integer](https://docs.python.org/3/reference/lexical_analysis.html#integer-literals)
and [floating-point](https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals)
literals, whose lexical definitions follow Python.

```
literal ::= integer | floatnumber
```

Literals are evaluated to Python values at compile time.

#### Parenthesized forms

```
parenth_form ::= "(" [expression_list] ")"
```

A parenthesized expression list is evaluated to whatever the expression list is
evaluated to. An empty pair of parentheses is evaluated to an empty tuple at
compile time.

#### Displays for lists and dictionaries

#### List displays

#### Dictionary displays

### Primaries

#### Attribute references

#### Subscriptions

#### Slicings

#### Calls

### The power operator

### Unary arithmetic and bitwise operations

### Binary arithmetic operations

### Shifting operations

### Binary bitwise operations

### Comparisons

#### Value comparisons

#### Membership test operations

#### Identity comparisons

### Boolean operations

### Conditional expressions

### Expression lists

```
expression_list ::= expression ("," expression)* [","]
```

Except when part of a list display, an expression list containing at least one
comma is evaluated to a tuple at compile time. The component expressions are
evaluated from left to right.

The trailing comma is required only to create a tuple with length 1; it is
optional in all other cases. A single expression without a trailing comma
is evaluated to the value of that expression.

## Simple statements

### Expression statements

### Assignment statements

#### Augmented assignment statements

#### Annotated assignment statements

### The `assert` statement
Assert statements are a convenient way to insert debugging assertions into a program:

```
assert_stmt ::=  "assert" expression ["," expression]
```

Assert statements are currently supported on the CPU, CUDA, and Metal backends.

Assert statements only work in debug mode (when `debug=True` is set in the arguments of `ti.init()`),
otherwise they are equivalent to no-op.

The simple form, `assert expression`, raises `TaichiAssertionError` (which is a subclass of `AssertionError`)
when `expression` is equal to `False`, with the code of `expression` as the error message.

The extended form, `assert expression1, expression2`, raises `TaichiAssertionError` when `expression1` is equal to `False`,
with `expression2` as the error message.

### The `pass` statement

### The `return` statement

### The `break` statement

### The `continue` statement

## Compound statements

### The `if` statement

### The `while` statement

### The `for` statement

The `for` statement in Taichi is used to iterate over a range of numbers, multidimensional ranges, or the indices of elements in a field.

```
for_stmt ::=  "for" target_list "in" expression ":" suite
```

Taichi does not support `else` clause in `for` statements.

The `for` loops can iterate in parallel if they are in the outermost scope.
When a `for` loop is parallelized, the order of iteration is not determined,
and it cannot be terminated by `break` statements.

Taichi uses `ti.loop_config` function to set directives for the loop right after it.
You can write `ti.loop_config(serialize=True)` before a range/ndrange `for` loop to let it run serially,
then it can be terminated by `break` statements.

There are four kinds of `for` statements:

- The range `for` statement iterates over a range of numbers.
- The ndrange `for` iterates over multidimensional ranges.
- The struct `for` statement iterates over every active elements in a taichi field.
- The static `for` statement unrolls a range/ndrange `for` at compile time.

#### The range `for` statement

The `expression` of range `for` statement must be like `range(begin, end)` or `range(end)`,
and they mean the same as the python `range` function,
except that the `step` argument is not supported.

The `target_list` of range `for` statement must be an identifier which
is not occupied in the current scope.

The range `for` loops are by default parallelized when the loops are in the outermost scope.

#### The ndrange `for` statement

The `expression` of ndrange `for` statement must be a call to `ti.ndrange()` or a nested call to `ti.grouped(ti.ndrange())`.
If the `expression` is a call to `ti.range()`, it is a normal ndrange `for`.
If the `expression` is a call to `ti.grouped(ti.range())`, it is a grouped ndrange `for`.


`ti.ndrange` receives arbitrary numbers of arguments.
The k-th argument represents the iteration range of the k-th dimension,
and the loop iterates over the [direct product](https://en.wikipedia.org/wiki/Direct_product) of the iteration range of each dimension.

Every argument must be an integer or a tuple of two integers.
If the k-th argument is an integer `n`, the range of the k-th dimension
is equivalent to the range of `range(n)` in python.
If the k-th argument is a tuple of two integers `(a, b)`, the range of the k-th dimension
is equivalent to the range of `range(a, b)` in python.

The `target_list` of an n-dimensional normal ndrange `for` statement must be n different identifiers which
are not occupied in the current scope, and the k-th identifier is assigned an integer which is the loop variable of the k-th dimension.

The `target_list` of an n-dimensional grouped ndrange `for` statement must be one identifier which
is not occupied in the current scope, and the identifier is assigned a `ti.Vector` with length n, which contains the loop variables of all n dimensions.

The ndrange `for` loops are by default parallelized when the loops are in the outermost scope.

#### The struct `for` statement
The `expression` of a struct `for` statement must be a Taichi field or a call to `ti.grouped(x)` where `x` is a Taichi field.

If the `expression` is a Taichi field, it is a normal struct `for`.
If the `expression` is a call to `ti.grouped(x)` where `x` is a Taichi field, it is a grouped struct `for`.

The `target_list` of a normal struct `for` statement on an n-dimensional field must be n different identifiers which
are not occupied in the current scope, and the k-th identifier is assigned an integer which is the loop variable of the k-th dimension.

The `target_list` of a grouped ndrange `for` statement on an n-dimensional field must be one identifier which
is not occupied in the current scope, and the identifier is assigned a `ti.Vector` with length n, which contains the loop variables of all n dimensions.

The struct `for` statement must be at the outermost scope of the kernel,
and it cannot be terminated by a `break` statement even when it is run serially.

#### The static `for` statement
The `expression` of a static `for` statement must be a call to `ti.static()` and the argument of `ti.static()` must meet the requirement on `expression` of the range/ndrange for, and
every variable inside `expression` must be a compile-time constant.

The static `for` statement unrolls the range/ndrange `for` loop at compile time.

For example,
```python
for i in ti.static(range(5)):
    print(i)
```
is unrolled to
```python
print(0)
print(1)
print(2)
print(3)
print(4)
```
at compile time.
