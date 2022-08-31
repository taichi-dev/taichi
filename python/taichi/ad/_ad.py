"""Taichi automatic differentiation module.

This module supplies two decorators for users to customize their
gradient computation task.
"""
import warnings
from functools import reduce

import numpy as np
from taichi.lang import impl
from taichi.lang.enums import SNodeGradType
from taichi.lang.field import ScalarField
from taichi.lang.snode import SNode

from taichi import _snode


class Tape:
    def __init__(self, loss=None, clear_gradients=True, validation=False):
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
            validation(Bool): Check whether the code inside the context manager is autodiff valid, e.g., agree with the global data access rule.

        Example::

            >>> @ti.kernel
            >>> def sum(a: ti.float32):
            >>>     for I in ti.grouped(x):
            >>>         y[None] += x[I] ** a
            >>>
            >>> with ti.ad.Tape(loss = y):
            >>>     sum(2)
        """
        self.calls = []
        self.entered = False
        self.gradient_evaluated = False
        self.clear_gradients = clear_gradients
        self.validation = validation
        self.runtime = impl.get_runtime()
        if not self.runtime.prog.config.debug and self.validation:
            warnings.warn(
                "Debug mode is disabled, autodiff valid check will not work. Please specify `ti.init(debug=True)` to enable the check.",
                Warning)
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
        if self.validation:
            clear_all_gradients(gradient_type=SNodeGradType.ADJOINT_CHECKBIT)

        from taichi._kernels import clear_loss  # pylint: disable=C0415
        clear_loss(self.loss)

        # Attach the context manager to runtime
        self.runtime.target_tape = self

    def __exit__(self, _type, value, tb):
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


def clear_all_gradients(gradient_type=SNodeGradType.ADJOINT):
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
                if ch.get_snode_grad_type() == gradient_type:
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
    system, e.g. `ti.ad.Tape()` and `kernel.grad()`.

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
    autodiff system, e.g. `ti.ad.Tape()` and `kernel.grad()`.
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


class FwdMode:
    def __init__(self, loss, param, seed=None, clear_gradients=True):
        self.calls = []
        self.entered = False
        self.kernels_recovered = False
        self.runtime = impl.get_runtime()
        self.loss = loss
        self.param = param
        self.seed = seed
        self.clear_gradients = clear_gradients

    def __enter__(self):
        assert not self.entered, "Forward mode manager can be entered only once."
        self.entered = True
        impl.get_runtime().materialize()
        if not isinstance(self.loss, list):
            self.loss = [self.loss]
        for ls in self.loss:
            assert isinstance(ls, ScalarField)

        # Currently we only support only one N-D field as a group of parameters,
        # which is sufficient for computing Jacobian-vector product(Jvp).
        # For cases with multiple groups of parameters, it requires to run the forward ad multiple times,
        # which is out of scope of the current design for this interface.

        # TODO: support vector field and matrix field
        assert isinstance(self.param, ScalarField)

        def shape_flatten(shape):
            return reduce((lambda x, y: x * y), list(shape))

        # Handle 0-D field
        if len(self.param.shape) != 0:
            parameters_shape_flatten = shape_flatten(self.param.shape)
        else:
            parameters_shape_flatten = 1

        if not self.seed:
            if parameters_shape_flatten == 1:
                # Compute the derivative respect to the first variable by default
                self.seed = [1.0]
            else:
                raise RuntimeError(
                    '`seed` is not set for non 0-D field, please specify.'
                    ' `seed` is a list to specify which parameters the computed derivatives respect to. The length of the `seed` should be same to that of the `parameters`'
                    ' E.g. Given a loss `loss = ti.field(float, shape=3)`, parameter `x = ti.field(float, shape=3)`'
                    '      seed = [0, 0, 1] indicates compute derivative respect to the third element of `x`.'
                    '      seed = [1, 1, 1] indicates compute the sum of derivatives respect to all three element of `x`, i.e., Jacobian-vector product(Jvp) for each element in `loss`'
                )
        else:
            assert parameters_shape_flatten == len(self.seed)

        # Clear gradients
        if self.clear_gradients:
            clear_all_gradients(gradient_type=SNodeGradType.DUAL)

        # Set seed for each variable
        if len(self.seed) == 1:
            if len(self.param.shape) == 0:
                # e.g., x= ti.field(float, shape = ())
                self.param.dual[None] = 1.0 * self.seed[0]
            else:
                # e.g., ti.root.dense(ti.i, 1).place(x.dual)
                self.param.dual[0] = 1.0 * self.seed[0]
        else:
            self.param.dual.from_numpy(np.array(self.seed, dtype=np.float32))

        # Attach the context manager to the runtime
        self.runtime.fwd_mode_manager = self

    def __exit__(self, _type, value, tb):
        self.runtime.fwd_mode_manager = None
        self.clear_seed()
        self.recover_kernels()

    def insert(self, func, mode_original):
        self.calls.append((func, mode_original))

    def recover_kernels(self):
        assert self.entered, "Before recover the kernels, fwd mode manager must be entered."
        for f, mode_original in self.calls:
            f.autodiff_mode = mode_original
        self.kernels_recovered = True

    def clear_seed(self):
        # clear seed values
        if len(self.seed) == 1:
            if len(self.param.shape) == 0:
                # e.g., x= ti.field(float, shape = ())
                self.param.dual[None] = 0.0
            else:
                # e.g., ti.root.dense(ti.i, 1).place(x.dual)
                self.param.dual[0] = 0.0
        else:
            self.param.dual.fill(0)


__all__ = [
    'FwdMode', 'Tape', 'clear_all_gradients', 'grad_for', 'grad_replaced',
    'no_grad'
]
