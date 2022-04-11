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

### Binary operation rule

For all binary operations, scalars are broadcast in the case of scalar-tensor/tensor-scalar operations, and tensor-tensor is done elementwise without any broadcasting. In tensor-tensor cases, all operands are required to be the same shape.

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

#### List and dictionary displays

Taichi supports
[displays](https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries)
for container (list and dictionary only) construction. Like in Python, a
display is one of:
- listing the container items explicitly;
- providing a *comprehension* (a set of looping and filtering instructions) to
compute the container items.

```
list_display       ::= "[" [expression_list | list_comprehension] "]"
list_comprehension ::= assignment_expression comp_for

dict_display       ::= "{" [key_datum_list | dict_comprehension] "}"
key_datum_list     ::= key_datum ("," key_datum)* [","]
key_datum          ::= expression ":" expression
dict_comprehension ::= key_datum comp_for

comp_for           ::= "for" target_list "in" or_test [comp_iter]
comp_iter          ::= comp_for | comp_if
comp_if            ::= "if" or_test [comp_iter]
```

The semantics of list and dict displays in Taichi mainly follow Python. Note
that they are evaluated at compile time, so all expressions in `comp_for`,
as well as keys in `key_datum`, are required to be evaluated to Python values.

For example, in the following code snippet, `a` can be successfully defined
while `b` cannot because `p` cannot be evaluated to a Python value at compile
time.

```python
@ti.kernel
def test(p: ti.i32):
    a = ti.Matrix([i * p for i in range(10)])  # valid
    b = ti.Matrix([i * p for i in range(p)])  # compile error
```

### Primaries

Primaries represent the most tightly bound operations.

```
primary ::= atom | attributeref | subscription | slicing | call
```

#### Attribute references

```
attributeref ::= primary "." identifier
```

Attribute references are evaluated at compile time. The `primary` must be
evaluated to a Python value with an attribute named `identifier`. Common use
cases in Taichi include metadata queries of
[field](https://docs.taichi.graphics/lang/articles/meta#field-metadata) and
[matrices](https://docs.taichi.graphics/lang/articles/meta#matrix--vector-metadata).

#### Subscriptions

```
subscription ::= primary "[" expression_list "]"
```

If `primary` is evaluated to a Python value (e.g., a list or a dictionary),
then all expressions in `expression_list` are required to be evaluated to
Python values, and the subscription is evaluated at compile time following
[Python](https://docs.python.org/3/reference/expressions.html#subscriptions).

Otherwise, `primary` has a Taichi type. All Taichi types excluding primitive
types support subscriptions. You can refer to documentation of these types
for subscription usage.

:::note
When `primary` has a Taichi matrix type, all expressions in `expression_list`
are required to be evaluated to Python values. This restriction can be got rid
of by setting `ti.init(dynamic_index=True)`.
:::

#### Slicings

```
slicing      ::= primary "[" slice_list "]"
slice_list   ::= slice_item ("," slice_item)* [","]
slice_item   ::= expression | proper_slice
proper_slice ::= [expression] ":" [expression] [ ":" [expression] ]
```

Currently, slicings are only supported when `primary` has a Taichi matrix type,
and the evaluation happens at compile time.
When `slice_item` is in the form of:
- a single `expression`: it is required to be evaluated to a Python value
unless `ti.init(dynamic_index=True)` is set.
- `proper_slice`: all expressions (the lower bound, the upper bound, and the
stride) inside have to be evaluated to Python values.

#### Calls

```
call                 ::= primary "(" [argument_list [","] | comprehension] ")"
argument_list        ::= positional_arguments ["," starred_and_keywords]
                           ["," keywords_arguments]
                         | starred_and_keywords ["," keywords_arguments]
                         | keywords_arguments
positional_arguments ::= positional_item ("," positional_item)*
positional_item      ::= assignment_expression | "*" expression
starred_and_keywords ::= ("*" expression | keyword_item)
                         ("," "*" expression | "," keyword_item)*
keywords_arguments   ::= (keyword_item | "**" expression)
                         ("," keyword_item | "," "**" expression)*
keyword_item         ::= identifier "=" expression


```

To favor simplicity, starred arguments and keyword arguments are not supported for Taichi kernels.

When calling a Python function, it needs to be wrapped by `ti.static()`.

The following Taichi builtin functions are supported: `print()`, `min()`, `max()`, `int()`, `float()`, `any()`, `all()`, `abs()`, `pow()`.

Calling a type annotation for literals

### The power operator

```
power ::= primary ["**" u_expr]
```

Applying '**' operator inside Taichi scope has the same behavior as Python. The power operator binds more tightly than unary operators on the left, but less tightly than unary operators on the right; i.e. -3 ** -2 == -(3 ** (-2)).

### Unary arithmetic and bitwise operations

```
u_expr ::= power | "-" power | "~" power
```
The unary - operator yields the negation of its argument. The unary ~ operator yields the bitwise inversion of its argument. - can be used with all scalar and tensor. ~ can only be used with integer scalar (i32, i64, etc.) and interrelated tensor.

### Binary arithmetic operations

```
m_expr ::= u_expr | m_expr "*" u_expr | m_expr "@" m_expr | m_expr "//" u_expr | m_expr "/" u_expr | m_expr "%" u_expr
a_expr ::= m_expr | a_expr "+" m_expr | a_expr "-" m_expr
```
The binary arithmetic operators can operate on scalar and tensor. For tensor-tensor ops, both arguments must have the same shape. For scalar-tensor or tensor-scalar ops, the scalar is usually broadcast to the size of the tensor. The @ operator is for matrix multiplication and only operates on Tensor arguments.

In division `/` operation, the compiler will automatically convert integral operands into default floating-point types (`f32` or `f64`).

When the operands have different types, the outcoming type will be set as the expected type under C++ (e.g. `i32` + `i64` = `i64`).

### Shifting operations

```
shift_expr::= a_expr | shift_expr ( "<<" | ">>" ) a_expr
```
These operators accept integer scalar (i32, i64, etc.) and interrelated tensor for both arguments. When both arguments are tensors, they must have the same shape. When one is a scalar and the other is a tensor, the scalar is logically broadcast to match the size of the tensor.

The compiler will check both operands to be integral type.

### Binary bitwise operations

```
and_expr ::= shift_expr | and_expr "&" shift_expr
xor_expr ::= and_expr | xor_expr "^" and_expr
or_expr  ::= xor_expr | or_expr "|" xor_expr
```
The & operator computes the bitwise AND of its arguments, the ^ the bitwise XOR, and the | the bitwise OR. The types requirements and broadcast logic are the same as shifting operations.

The compiler will check both operands to be integral type.

### Comparisons

```
comparison    ::= or_expr (comp_operator or_expr)*
comp_operator ::= "<" | ">" | "==" | ">=" | "<=" | "!=" | "is" ["not"] | ["not"] "in"
```
A comparison yields a boolean value (True or False), or if one of the operands is a Tensor, a boolean Tensor. Comparisons can be chained arbitrarily as long as they do not yield boolean Tensors that have more than one element. a op1 b op2 c ... is equivalent to a op1 b and b op2 c and ....

All operands need to be primaries. The return type is `i32` for all comparision operations.

#### Value comparisons

The operators <, >, ==, >=, <=, and != compare the values of two objects. The two objects generally need to be of the same type, unless there is an implicit type conversion available between the objects. Built-in Python types like strings, lists, tuples are not supported. They need to be converted into Taichi scalar or tensor in advance.

#### Membership test operations

List, dict and tuple types along with 'in' operations, are not supported in Taichi scope. The only exception is to put the whole expression into 'ti.static()' which degenerates into Python expression. More details refer to Python's specifications.
### Boolean operations

```
or_test  ::= and_test | or_test "or" and_test
and_test ::= not_test | and_test "and" not_test
not_test ::= comparison | "not" not_test
```

When the operator is inside a [static expression](#static-expressions),
the evaluation rule of the operator follows [Python](https://docs.python.org/3/reference/expressions.html#boolean-operations).
Otherwise, the behavior depends on the `short_circuit_operators` option of `ti.init()`:
- If `short_circuit_operators` is `False` (default), a *logical and* will be
treated as a *bitwise AND*, and a *logical or* will be treated as a *bitwise
OR*. See [binary bitwise operations](#binary-bitwise-operations) for details.
- If `short_circuit_operators` is `True`, the normal short circuiting behavior
is adopted, and the operands are required to be boolean values. Since Taichi
does not have boolean type yet, `ti.i32` is served as a temporary alternative.
A `ti.i32` value is considered `False` if and only if the value is evaluated to 0.

### Assignment expressions

```
assignment_expression ::= [identifier ":="] expression
```

An assignment expression assigns an expression to an identifier (see
[assignment statements](#assignment-statements) for more details),
while also returning the value of the expression.

Example:
```python
@ti.kernel
def foo() -> ti.i32:
    b = 2 + (a := 5)
    b += a
    return b
# the return value should be 12
```

:::note
This operator is supported since Python 3.8.
:::

### Conditional expressions

### Static expressions

```
static_expression ::= "ti.static(" positional_arguments ")"
```

Static expressions are expressions that are wrapped by a call to `ti.static()`.
The `positional_arguments` is evaluated at compile time, and the items inside must be evaluated to Python values.

`ti.static()` receives one or more arguments.
- When a single argument is passed in, it returns the argument.
- When multiple arguments are passed in, it returns a tuple containing all the arguments in the same order as they are passed.

The static expressions work as a mechanism to trigger many metaprogramming functions in Taichi,
such as [compile-time loop unrolling and compile-time branching](lang/articles/advanced/meta.md#compile-time-evaluations).

The static expressions can also be used to [create aliases for Taichi fields and Taichi functions](lang/articles/advanced/syntax_sugars.md#aliases).

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
assert_stmt ::= "assert" expression ["," expression]
```

Assert statements are currently supported on the CPU, CUDA, and Metal backends.

Assert statements only work in debug mode (when `debug=True` is set in the arguments of `ti.init()`),
otherwise they are equivalent to no-op.

The simple form, `assert expression`, raises `TaichiAssertionError` (which is a subclass of `AssertionError`)
when `expression` is equal to `False`, with the code of `expression` as the error message.

The extended form, `assert expression1, expression2`, raises `TaichiAssertionError` when `expression1` is equal to `False`,
with `expression2` as the error message. `expression2` must be a constant or a formatted string. The variables in the
formatted string must be scalars.

### The `pass` statement
```
pass_stmt ::= "pass"
```

`pass` is a null operation — when it is executed, nothing happens.
It is useful as a placeholder when a statement is required syntactically, but no code needs to be executed.

### The `return` statement
```
return_stmt ::= "return" [expression_list]
```

The return statement may only occur once in a Taichi kernel or a Taichi function,
and it must be at the bottom of the function body.
Note that this is subject to change, and Taichi might relax it in the future.

If a Taichi kernel or Taichi function has a return type hint,
it must have a return statement that returns a value other than `None`.

If a Taichi kernel has a return statement that returns a value other than `None`, it must have a return type hint.
The return type hint for Taichi function is optional but recommended.
Note that this is subject to change, and Taichi might enforce it in the future.

A kernel can have at most one return value, which can be a scalar, `ti.Matrix`, or `ti.Vector`,
and the number of elements in the return value must not exceed 30.
Note that this number is an implementation detail, and Taichi might relax it in the future.

A Taichi function can have multiple return values in a return statement,
and the return values can be scalar, `ti.Vector`, `ti.Matrix`, `ti.Struct`, and more.

### The `break` statement
```
break_stmt ::= "break"
```

The break statement may only occur syntactically nested in a for or while loop, and it terminates the nearest enclosing loop.

Break statement is not allowed when the nearest enclosing loop is a parallel range/ndrange for loop,
a struct for loop, or a mesh for loop.

### The `continue` statement
```
continue_stmt ::= "continue"
```

The continue statement may only occur syntactically nested in a for or while loop,
and it continues with the next cycle of the nearest enclosing loop.

## Compound statements

### The `if` statement

### The `while` statement

### The `for` statement
