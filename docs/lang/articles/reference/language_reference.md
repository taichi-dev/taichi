---
sidebar_position: 1
---

# Language Reference

This article describes the syntax and semantics of the Taichi programming
language.

**To users**: If you have gone through the user tutorials and still feel uncertain
about your program behavior, then you are in the right place. If you find the
actual behavior different from what is described in this article, feel free to
create an [issue](https://github.com/taichi-dev/taichi/issues/new/choose).
You should not rely solely on this article since things unspecified are subject to changes.

**To contributors**: This article specifies what the language *should* be. That
is, you should try to match the implementation of the Taichi compiler with this
article. You can clearly determine a certain behavior is *correct*, *buggy*, or
*undefined* from this article.

## Introduction

Taichi is a domain-specific language embedded in Python.
[Kernels and functions](../kernels/syntax.md) clearly defines the boundary between
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
interaction with Python and [meta-programming](../advanced/meta.md) support, there
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
- A primitive type, as described in [Type system](../type/type.md)
- A compound type, as described in [Type system](../type/type.md)
- An ndarray type, as introduced in [Run a Taichi Program using Ndarray on
Android](../deployment/ndarray_android.md)
- A sparse matrix builder type, as introduced in [Sparse
Matrix](../math/sparse_matrix.md)

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

### Common rules of binary operations

Following the [Values and types](#values-and-types) section, if both operands
of a binary operation are Python values, compile-time evaluation is triggered
and a result Python value is produced. If only one operand is a Python value,
it is first turned into a Taichi value with
[default type](../type/type.md#default-primitive-types-for-integers-and-floating-point-numbers).
Now the only remaining case is that both operands are Taichi values.

Binary operations can happen between Taichi values of either primitive type or
compound type. There are three cases in total:
- Two primitive type values. The return type is also a primitive type.
- One primitive type value and one compound type value. The primitive type
value is first broadcast into the shape of the compound type value. Now it
belongs to the case of two compound type values.
- Two compound type values. For operators other than matrix multiplication,
both values are required to have the same shape, and the operator is conducted
element-wise, resulting in a compound type value with same shape.

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
[field](../advanced/meta.md#field-metadata) and
[matrices](../advanced/meta.md#matrix--vector-metadata).

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
call                 ::= primary "(" [positional_arguments] ")"
positional_arguments ::= positional_item ("," positional_item)*
positional_item      ::= assignment_expression | "*" expression
```

The `primary` must be evaluated to one of:
- A [Taichi function](../kernels/syntax.md#taichi-function).
- A [Taichi builtin function](./operator.md#other-arithmetic-functions).
- A Taichi primitive type. In this case, the `positional_arguments` must only
  contain one item. If the item is evaluated to a Python value, then the
  primitive type serves as a type annotation for a literal, and the Python value
  will be turned into a Taichi value with that annotated type. Otherwise, the
  primitive type serves as a syntax sugar for `ti.cast()`, but the item cannot
  have a compound type.
- A Python callable object. If not inside a [static expression](#static-expressions), a warning is produced.

### The power operator

```
power ::= primary ["**" u_expr]
```

The power operator has the same semantics as the builtin `pow()` function.

### Unary arithmetic and bitwise operations

```
u_expr ::= power | "-" power | "+" power | "~" power
```

Similar to [rules for binary operations](#common-rules-of-binary-operations),
if the operand is a Python value, compile-time evaluation is triggered and a
result Python value is produced. Now the remaining case is that the operand is
a Taichi value:
- If the operand is a primitive type value, the return type is also a primitive
type.
- If the operand is a compound type value, the operator is conducted
element-wise, resulting in a compound type value with same shape.

See [arithmetic operators](./operator.md#arithmetic-operators) and
[bitwise operators](./operator.md#bitwise-operators) for operator details.
Note that `~` can only be used with integer type values.

### Binary arithmetic operations

```
m_expr ::= u_expr | m_expr "*" u_expr | m_expr "@" m_expr | m_expr "//" u_expr | m_expr "/" u_expr | m_expr "%" u_expr
a_expr ::= m_expr | a_expr "+" m_expr | a_expr "-" m_expr
```

See [common rules for binary operations](#common-rules-of-binary-operations),
[implicit type casting in binary operations](../type/type.md#implicit-type-casting-in-binary-operations),
and [arithmetic operators](./operator.md#arithmetic-operators). Note that
the `@` operator is for matrix multiplication and only operates on matrix type
arguments.
### Shifting operations

```
shift_expr::= a_expr | shift_expr ( "<<" | ">>" ) a_expr
```

See [common rules for binary operations](#common-rules-of-binary-operations),
[implicit type casting in binary operations](../type/type.md#implicit-type-casting-in-binary-operations),
and [bitwise operators](./operator.md#bitwise-operators). Note that both operands
are required to have integer types.

### Binary bitwise operations

```
and_expr ::= shift_expr | and_expr "&" shift_expr
xor_expr ::= and_expr | xor_expr "^" and_expr
or_expr  ::= xor_expr | or_expr "|" xor_expr
```

See [common rules for binary operations](#common-rules-of-binary-operations),
[implicit type casting in binary operations](../type/type.md#implicit-type-casting-in-binary-operations),
and [bitwise operators](./operator.md#bitwise-operators). Note that both operands
are required to have integer types.

### Comparisons

```
comparison    ::= or_expr (comp_operator or_expr)*
comp_operator ::= "<" | ">" | "==" | ">=" | "<=" | "!=" | ["not"] "in"
```

Comparisons can be chained arbitrarily, e.g., `x < y <= z` is equivalent to `(x < y) & (y <= z)`.

#### Value comparisons

See [common rules for binary operations](#common-rules-of-binary-operations),
[implicit type casting in binary operations](../type/type.md#implicit-type-casting-in-binary-operations),
and [comparison operators](./operator.md#comparison-operators).

#### Membership test operations

The semantics of membership test operations follow
[Python](https://docs.python.org/3/reference/expressions.html#membership-test-operations),
but they are only supported in [static expressions](#static-expressions).
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

```
conditional_expression ::= or_test ["if" or_test "else" expression]
expression             ::= conditional_expression
```

The expression `x if C else y` first evaluates the condition, `C` rather than `x`.
If `C` is `True` (the meaning of `True` and `False` has been mentioned at [boolean operations](#boolean-operations)), `x` is evaluated and its value is returned; otherwise,`y` is evaluated and its value is returned.

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
such as [compile-time loop unrolling and compile-time branching](../advanced/meta.md#compile-time-evaluations).

The static expressions can also be used to [create aliases for Taichi fields and Taichi functions](./syntax_sugars.md#aliases).

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

This section explains the syntax and semantics of compound statements in Taichi. A simple statement is comprised within a single logical line. Several simple statements may occur on a single line separated by semicolons.

```
simple_stmt ::= expression_stmt
                | assert_stmt
                | assignment_stmt
                | augmented_assignment_stmt
                | annotated_assignment_stmt
                | pass_stmt
                | return_stmt
                | break_stmt
                | continue_stmt
```


### Expression statements

```
expression_stmt    ::= expression_list
```

An expression statement evaluates the expression list (which may be a single expression).

### Assignment statements

```
assignment_stmt ::= (target_list "=")+ expression_list
target_list     ::= target ("," target)* [","]
target          ::= identifier
                    | "(" [target_list] ")"
                    | "[" [target_list] "]"
                    | attributeref
                    | subscription
```

The recursive definition of an assignment statement basically follows
[Python](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements),
with the following points to notice:
- According to the [Variables and scope](#variables-and-scope) section, if a
target is an identifier appearing for the first time, a variable is defined
with that name and inferred type from the corresponding right-hand side
expression. If the expression is evaluated to a Python value, it will be turned
into a Taichi value with [default type](../type/type.md#default-primitive-types-for-integers-and-floating-point-numbers).
- If a target is an existing identifier, the corresponding right-hand side
expression must be evaluated to a Taichi value with the type of the
corresponding variable of that identifier. Otherwise, an implicit cast will
happen.

#### Augmented assignment statements

```
augmented_assignment_stmt ::= augtarget augop expression_list
augtarget                 ::= identifier | attributeref | subscription
augop                     ::= "+=" | "-=" | "*=" | "/=" | "//=" | "%=" |
                              "**="| ">>=" | "<<=" | "&=" | "^=" | "|="
```

Different from [Python](https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements), some augmented assignments (e.g., `x[i] += 1`) are [automatically atomic](./operator.md#supported-atomic-operations) in Taichi.

#### Annotated assignment statements

```
annotated_assignment_stmt ::= identifier ":" expression "=" expression
```
The differences from normal [assignment statements](#assignment-statements) are:
- Only single identifier target is allowed.
- If the identifier appears for the first time, a variable is defined
with that name and type annotation (the expression after ":"). The right-hand
side expression is cast to a Taichi value with the annotated type.
- If the identifier already exists, the type annotation must be the same as the
type of the corresponding variable of the identifier.

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

This section explains the syntax and semantics of compound statements in Taichi.

A compound statement consists of one or more *clauses*.
A *clause* consists of a header and a *suite*.
The *clause headers* of a particular compound statement are all at the same indentation level.
Each *clause header* begins with a uniquely identifying keyword and ends with a colon.
A *suite* is a group of statements controlled by a *clause*.

```
compound_stmt ::= if_stmt | while_stmt | for_stmt
suite         ::= stmt_list NEWLINE | NEWLINE INDENT statement+ DEDENT
statement     ::= stmt_list NEWLINE | compound_stmt
stmt_list     ::= simple_stmt (";" simple_stmt)* [";"]
```

The difference between the compound statements in Taichi and Python is that Taichi introduces
compile time evaluation. If the expression in the *clause header* is a static expression,
Taichi replaces the compound statement at compile time according to the evaluation result of the expression.

### The `if` statement

The `if` statement is used for conditional execution:
```
if_stmt ::= "if" (static_expression | assignment_expression) ":" suite
            ("elif" (static_expression | assignment_expression) ":" suite)*
            ["else" ":" suite]
```

The `elif` *clause* is a syntax sugar for a `if` statement inside a `else` *clause*.
For example:

```python
if cond_a:
    body_a
elif cond_b:
    body_b
elif cond_c:
    body_c
else:
    body_d
```
is equivalent to
```python
if cond_a:
    body_a
else:
    if cond_b:
        body_b
    else:
        if cond_c:
            body_c
        else:
            body_d
```
Taichi first transforms `elif` *clause* as above, and then deal with the `if` statement with only an `if` *clause* and possibly an `else` *clause* as below.

If the expression of the `if` *clause* is found to be true (see section [Boolean operations](#boolean-operations) for the definition of true and false),
the *suite* of the `if` *clause* is executed. Otherwise, the *suite* of the `else` *clause*, if present, is executed.

An `if` statement whose expression is a static expression is called a static `if` statement.
The expression of a static `if` *clause* is evaluated at compile time, and it replaces the compound statement as below at compile time.
- If the static expression is found to be true, the *suite* of the `if` *clause* replaces the static `if` statement.
- If the static expression is found to be false, and there is an `else` *clause*, the *suite* of the `else` *clause* replaces the static `if` statement.
- If the static expression is found to be false, and there is no `else` *clause*, a `pass` statement replaces the static `if` statement.

### The `while` statement

The `while` statement is used for repeated execution as long as an expression is true:
```
while_stmt ::= "while" assignment_expression ":" suite
```

This repeatedly tests the expression and, if it is true, executes the *suite*;
if the expression is false (which may be the first time it is tested) the loop terminates.

A [`break` statement](#the-break-statement) executed in the *suite* terminates the loop.
A [`continue` statement](#the-continue-statement) executed in the *suite* skips the rest of the *suite* and
goes back to testing the expression.

### The `for` statement

The `for` statement in Taichi is used to iterate over a range of numbers, multidimensional ranges, or the indices of elements in a field.

```
for_stmt        ::= "for" target_list "in" iter_expression ":" suite
iter_expression ::= static_expression | expression
```

Taichi does not support `else` clause in `for` statements.

The `for` loops can iterate in parallel if they are in the outermost scope.
When a `for` loop is parallelized, the order of iteration is not determined,
and it cannot be terminated by `break` statements.

Taichi uses `ti.loop_config` function to set directives for the loop right after it.
You can write `ti.loop_config(serialize=True)` before a range/ndrange `for` loop to let it run serially,
then it can be terminated by `break` statements.

There are four kinds of `for` statements:

- The range `for` statement
- The ndrange `for` statement
- The struct `for` statement
- The static `for` statement

#### The range `for` statement

The range `for` statement iterates over a range of numbers.

The `iter_expression` of range `for` statement must be like `range(start, stop)` or `range(stop)`,
and they mean the same as [the Python `range` function](https://docs.python.org/3/library/stdtypes.html#range),
except that the `step` argument is not supported.

The `target_list` of range `for` statement must be an identifier which
is not occupied in the current scope.

The range `for` loops are by default parallelized when the loops are in the outermost scope.

#### The ndrange `for` statement

The ndrange `for` iterates over multidimensional ranges.

The `iter_expression` of ndrange `for` statement must be a call to `ti.ndrange()` or a nested call to `ti.grouped(ti.ndrange())`.
- If the `iter_expression` is a call to `ti.range()`, it is a normal ndrange `for`.
- If the `iter_expression` is a call to `ti.grouped(ti.range())`, it is a grouped ndrange `for`.

You can use grouped `for` loops to write [dimensionality-independent programs](../advanced/meta.md#dimensionality-independent-programming-using-grouped-indices).

`ti.ndrange` receives arbitrary numbers of arguments.
The k-th argument represents the iteration range of the k-th dimension,
and the loop iterates over the [direct product](https://en.wikipedia.org/wiki/Direct_product) of the iteration range of each dimension.

Every argument must be an integer or a tuple of two integers.
- If the k-th argument is an integer `stop`, the range of the k-th dimension
is equivalent to the range of `range(stop)` in Python.
- If the k-th argument is a tuple of two integers `(start, stop)`, the range of the k-th dimension
is equivalent to the range of `range(start, stop)` in Python.

The `target_list` of an n-dimensional normal ndrange `for` statement must be n different identifiers which
are not occupied in the current scope, and the k-th identifier is assigned an integer which is the loop variable of the k-th dimension.

The `target_list` of an n-dimensional grouped ndrange `for` statement must be one identifier which
is not occupied in the current scope, and the identifier is assigned a `ti.Vector` with length n, which contains the loop variables of all n dimensions.

The ndrange `for` loops are by default parallelized when the loops are in the outermost scope.

#### The struct `for` statement

The struct `for` statement iterates over every active elements in a Taichi field.

The `iter_expression` of a struct `for` statement must be a Taichi field or a call to `ti.grouped(x)` where `x` is a Taichi field.

- If the `iter_expression` is a Taichi field, it is a normal struct `for`.
- If the `iter_expression` is a call to `ti.grouped(x)` where `x` is a Taichi field, it is a grouped struct `for`.

The `target_list` of a normal struct `for` statement on an n-dimensional field must be n different identifiers which
are not occupied in the current scope, and the k-th identifier is assigned an integer which is the loop variable of the k-th dimension.

The `target_list` of a grouped struct `for` statement on an n-dimensional field must be one identifier which
is not occupied in the current scope, and the identifier is assigned a `ti.Vector` with length n, which contains the loop variables of all n dimensions.

The struct `for` statement must be at the outermost scope of the kernel,
and it cannot be terminated by a `break` statement even when it is run serially.

#### The static `for` statement

The static `for` statement unrolls a range/ndrange `for` loop at compile time.

If the `iter_expression` of the `for` statement is a [`static_expression`](#static-expressions),
the `for` statement is a static `for` statement.

The `positional_arguments` of the `static_expression` must meet the requirement on
`iter_expression` of the range/ndrange for.

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
