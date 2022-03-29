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

### Naming and scope

## Expressions

### Atoms

#### Identifiers (Names)

#### Literals

#### Parenthesized forms

#### Displays for lists, sets and dictionaries

#### List displays

#### Set displays

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

### Assignment expressions

### Conditional expressions

### Expression lists

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

Assert statements are effective only when `debug=True` is specified in the arguments of `ti.init()`,
otherwise they are equivalent to no-op.

The simple form, `assert expression`, raises `TaichiAssertionError` when `expression` is equal to `False`,
with the code of `expression` as the error message.

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
