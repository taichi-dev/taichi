from taichi.lang import impl


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
