"""Taichi automatic differentiation module.

This module supplies two decorators for users to customize their
gradient computation task.
"""
from taichi.lang import impl
from taichi.lang.snode import SNode

from taichi import _snode


class Tape:
    def __init__(self, loss=None, clear_gradients=True):
        """A context manager for reverse mode autodiff :class:`~taichi.ad.Tape`. The
        context manager would catching all of the callings of functions that
        decorated by :func:`~taichi.lang.kernel_impl.kernel` or
        :func:`~taichi.ad.grad_replaced` under `with` statement, and calculate
        all the partial gradients of a given loss variable by calling all of the
        gradient function of the callings caught in reverse order while `with`
        statement ended.

        See also :func:`~taichi.lang.kernel_impl.kernel` and
        :func:`~taichi.ad.grad_replaced` for gradient functions.

        Args:
            loss(:class:`~taichi.lang.expr.Expr`): The loss field, which shape should be ().
            clear_gradients(Bool): Before `with` body start, clear all gradients or not.

        Example::

            >>> @ti.kernel
            >>> def sum(a: ti.float32):
            >>>     for I in ti.grouped(x):
            >>>         y[None] += x[I] ** a
            >>>
            >>> with ti.Tape(loss = y):
            >>>     sum(2)
        """
        self.calls = []
        self.entered = False
        self.gradient_evaluated = False
        self.clear_gradients = clear_gradients
        self.runtime = impl.get_runtime()
        self.eval_on_exit = loss is not None
        self.loss = loss

    def __enter__(self):
        assert not self.entered, "Tape can be entered only once."
        self.entered = True

        impl.get_runtime().materialize()
        if len(self.loss.shape) != 0:
            raise RuntimeError(
                'The loss of `Tape` must be a 0-D field, i.e. scalar')
        if not self.loss.snode.ptr.has_adjoint():
            raise RuntimeError(
                'Gradients of loss are not allocated, please use ti.field(..., needs_grad=True)'
                ' for all fields that are required by autodiff.')
        if self.clear_gradients:
            clear_all_gradients()

        from taichi._kernels import clear_loss  # pylint: disable=C0415
        clear_loss(self.loss)
        self.runtime.target_tape = self

    def __exit__(self, _type, value, tb):
        # print('# kernel calls', len(self.calls))
        self.runtime.target_tape = None
        if self.eval_on_exit:
            self.grad()

    def insert(self, func, args):
        self.calls.append((func, args))

    def grad(self):
        assert self.entered, "Before evaluating gradients tape must be entered."
        assert not self.gradient_evaluated, "Gradients of grad can be evaluated only once."
        for func, args in reversed(self.calls):
            func.grad(*args)
        self.gradient_evaluated = True


def clear_all_gradients():
    """Sets the gradients of all fields to zero.
    """
    impl.get_runtime().materialize()

    def visit(node):
        places = []
        for _i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(_i)
            if not ch.is_place():
                visit(SNode(ch))
            else:
                if not ch.is_primal():
                    places.append(ch.get_expr())

        places = tuple(places)
        if places:
            from taichi._kernels import \
                clear_gradients  # pylint: disable=C0415
            clear_gradients(places)

    for root_fb in _snode.FieldsBuilder._finalized_roots():
        visit(root_fb)


def grad_replaced(func):
    """A decorator for python function to customize gradient with Taichi's autodiff
    system, e.g. `ti.Tape()` and `kernel.grad()`.

    This decorator forces Taichi's autodiff system to use a user-defined gradient
    function for the decorated function. Its customized gradient must be decorated
    by :func:`~taichi.ad.grad_for`.

    Args:
        fn (Callable): The python function to be decorated.

    Returns:
        Callable: The decorated function.

    Example::

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
        >>> @ti.grad_replaced
        >>> def foo(a):
        >>>     multiply(a)
        >>>
        >>> @ti.grad_for(foo)
        >>> def foo_grad(a):
        >>>     multiply_grad(a)"""
    def decorated(*args, **kwargs):
        # TODO [#3025]: get rid of circular imports and move this to the top.
        impl.get_runtime().grad_replaced = True
        if impl.get_runtime().target_tape:
            impl.get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            impl.get_runtime().grad_replaced = False

    decorated.grad = None
    return decorated


def grad_for(primal):
    """Generates a decorator to decorate `primal`'s customized gradient function.

    See :func:`~taichi.lang.grad_replaced` for examples.

    Args:
        primal (Callable): The primal function, must be decorated by :func:`~taichi.ad.grad_replaced`.

    Returns:
        Callable: The decorator used to decorate customized gradient function."""
    def decorator(func):
        def decorated(*args, **kwargs):
            func(*args, **kwargs)

        if not hasattr(primal, 'grad'):
            raise RuntimeError(
                f'Primal function `{primal.__name__}` must be decorated by ti.ad.grad_replaced'
            )
        if primal.grad is not None:
            raise RuntimeError(
                'Primal function must be a **python** function instead of a taichi kernel. Please wrap the taichi kernel in a @ti.ad.grad_replaced decorated python function instead.'
            )
        primal.grad = decorated
        return decorated

    return decorator


def no_grad(func):
    """A decorator for python function to skip gradient calculation within Taichi's
    autodiff system, e.g. `ti.Tape()` and `kernel.grad()`.
    This decorator forces Taichi's autodiff system to use an empty gradient function
    for the decorated function.

    Args:
        fn (Callable): The python function to be decorated.

    Returns:
        Callable: The decorated function.

    Example::

        >>> @ti.kernel
        >>> def multiply(a: ti.float32):
        >>>     for I in ti.grouped(x):
        >>>         y[I] = x[I] * a
        >>>
        >>> @ti.no_grad
        >>> def foo(a):
        >>>     multiply(a)"""
    def decorated(*args, **kwargs):
        impl.get_runtime().grad_replaced = True
        if impl.get_runtime().target_tape:
            impl.get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            impl.get_runtime().grad_replaced = False

    def placeholder(*args, **kwargs):
        return

    decorated.grad = placeholder
    return decorated
