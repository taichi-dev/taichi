"""Taichi automatic differentiation module.

This module supplies two decorators for users to customize their
gradient computation task.
"""
import warnings
from functools import reduce

import numpy as np
import taichi.types.primitive_types as types
from taichi.lang import impl
from taichi.lang.enums import AutodiffMode, SNodeGradType
from taichi.lang.expr import Expr
from taichi.lang.field import ScalarField
from taichi.lang.kernel_impl import kernel
from taichi.lang.snode import SNode
from taichi.types import ndarray, template

from taichi import _snode


class GradChecker:
    def __init__(self, loss, to_check):
        self.to_check = to_check
        self.loss = loss
        self.eps_range = 2.**np.arange(-3, -30, -2).astype(np.float64)
        self.result = [None] * len(to_check)
        self.all_fields = get_all_fields()
        self.backups = save_all_fields(self.all_fields)

    def add_calls(self, calls):
        self.calls = calls

    def check_grad(self):
        assert self.loss.dtype == types.f64, "Only f64 is supported when checking grad."

        @kernel
        def x_pos(x: template(), tangent_np: ndarray(), eps: types.f64):
            for I in impl.grouped(x):
                x[I] += eps * tangent_np[I]

        @kernel
        def x_neg(x: template(), tangent_np: ndarray(), eps: types.f64):
            for I in impl.grouped(x):
                x[I] -= eps * tangent_np[I]

        for i, x in enumerate(self.to_check):
            if x is self.loss:
                self.result[i] = True
                continue

            check_pass = False

            re_range = []
            for eps in self.eps_range:
                tangent_np = np.array(np.random.rand(*x.shape)).astype(
                    np.float64)

                restore_all_fields(self.all_fields, self.backups)
                x_pos(x, tangent_np, eps)
                for func, args in self.calls:
                    func(*args)
                loss_pos = self.loss.to_numpy()

                restore_all_fields(self.all_fields, self.backups)
                x_neg(x, tangent_np, eps)
                for func, args in self.calls:
                    func(*args)
                loss_neg = self.loss.to_numpy()

                ip_numerical = (loss_pos - loss_neg) * 0.5 / eps
                x_grad_np = x.grad.to_numpy()
                extra_dim = x_grad_np.ndim - tangent_np.ndim
                if extra_dim == 1:
                    tangent_np = np.expand_dims(tangent_np, axis=-1)
                if extra_dim == 2:
                    tangent_np = np.expand_dims(tangent_np, axis=-1)

                ip_autodiff = np.sum(x_grad_np * tangent_np)
                err = abs(ip_autodiff - ip_numerical)
                if ip_numerical != 0:
                    re = err / abs(ip_autodiff)
                else:
                    re = err / (abs(ip_autodiff) + 1e-20)
                re_range.append(re)

                if err * 100 <= abs(ip_autodiff):
                    check_pass = True
                    break

            self.result[i] = check_pass

            if not check_pass:
                print("variable", i, "has relative error", min(re_range),
                      ", expected relative error 0.01")
            else:
                print("variable", i, "passes grad check")

        assert all(self.result
                   ), "Grad check failed: Not all variables pass grad check"

        restore_all_fields(self.all_fields, self.backups)
        for func, args in self.calls:
            func(*args)


def get_all_fields():
    def visit(node, fields):
        for _i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(_i)
            if not ch.is_place():
                visit(SNode(ch), fields)
            else:
                if not ch.is_primal():
                    continue
                fields.append(ScalarField(Expr(ch.get_expr())))

    fields = []
    for root_fb in _snode.FieldsBuilder._finalized_roots():
        visit(root_fb, fields)
    return fields


def save_all_fields(all_fields):
    return [x.to_numpy() for x in all_fields]


def restore_all_fields(all_fields, backups):
    assert len(all_fields) == len(backups)
    for f, x in zip(all_fields, backups):
        f.from_numpy(x)


class Tape:
    def __init__(self,
                 loss=None,
                 clear_gradients=True,
                 validation=False,
                 grad_check=None,
                 checkpointer=None,
                 enable_checkpointing=True):
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
            grad_check(List[Field]): List of fields that need to check gradients.

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
        self.modes = []
        self.entered = False
        self.gradient_evaluated = False
        self.checkpointer = checkpointer
        # TODO: assert checkpointer is a CheckpointerManager
        if not self.checkpointer:
            global _checkpointer
            self.checkpointer = _checkpointer
        self.enable_checkpointing = enable_checkpointing
        self.calls_count = 0
        self.clear_gradients = clear_gradients
        self.validation = validation
        self.runtime = impl.get_runtime()
        if not self.runtime.prog.config().debug and self.validation:
            warnings.warn(
                "Debug mode is disabled, autodiff valid check will not work. Please specify `ti.init(debug=True)` to enable the check.",
                Warning)
        self.eval_on_exit = loss is not None
        self.loss = loss
        self.grad_checker = None
        if grad_check:
            assert isinstance(
                grad_check, list
            ), "grad_check should be a list of fields that need to check gradients."
            self.grad_checker = GradChecker(loss, grad_check)

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

        if self.checkpointer and self.enable_checkpointing:
            self.checkpointer.reset()

        # Attach the context manager to runtime
        self.runtime.target_tape = self
        return self

    def __exit__(self, _type, value, tb):
        self.runtime.target_tape = None
        # Save the computed forward result both in enable/disable checkpointing mode
        assert self.checkpointer, "Checkpointer is not initialized."
        self.checkpointer.save_forward_results()

        if self.eval_on_exit:
            self.grad()

        # Restore the computed forward result both in enable/disable checkpointing mode
        assert self.checkpointer, "Checkpointer is not initialized."
        self.checkpointer.restore_forward_results()
        self.checkpointer.clear()

        for calls, mode in zip(self.calls, self.modes):
            calls[0].autodiff_mode = mode

    def insert(self, func, args):
        # Kernels with mode `AutodiffMode.NONE` and `AutodiffMode.VALIDATION` are all forward kernels.
        # The difference is there are `assert` for global data access rule check in VALIDATION kernels.
        assert func.autodiff_mode in (
            AutodiffMode.NONE, AutodiffMode.VALIDATION
        ), "Inserted funcs should be forward kernels."
        self.modes.append(func.autodiff_mode)
        if self.validation:
            func.autodiff_mode = AutodiffMode.VALIDATION
        self.calls.append((func, args))

        if self.checkpointer and self.enable_checkpointing:
            self.calls_count += 1
            # Save a checkpoint before the kernel launch
            self.checkpointer.save_primal(self.calls_count)
            # print("calls count increase ", self.calls_count)

    def grad(self):
        assert self.entered, "Before evaluating gradients tape must be entered."
        assert not self.gradient_evaluated, "Gradients of grad can be evaluated only once."

        if self.checkpointer:
            self.checkpointer.get_forward_calls(self.calls)

        for func, args in reversed(self.calls):

            if self.checkpointer and self.enable_checkpointing:
                # Restore the checkpoint before launch the grad kernel
                # print("calls count ", self.calls_count)
                self.checkpointer.restore_primal(self.calls_count)
            # TODO: how to preserve the value of the seed?
            self.loss.grad.fill(1.0)
            # we need to check whether "func" has "grad" attribute
            # since we insert write_int and write_float kernels to self.calls
            # e.g. x[None] = 0.0, this func has no grad attribute
            if hasattr(func, 'grad'):
                func.grad(*args)
            if self.checkpointer and self.enable_checkpointing:
                self.calls_count -= 1
        self.gradient_evaluated = True
        if self.grad_checker:
            self.grad_checker.add_calls(self.calls)
            self.grad_checker.check_grad()


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
        >>> @ti.ad.grad_replaced
        >>> def foo(a):
        >>>     multiply(a)
        >>>
        >>> @ti.ad.grad_for(foo)
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
    decorated.autodiff_mode = AutodiffMode.NONE
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
    decorated.autodiff_mode = AutodiffMode.NONE
    return decorated


class FwdMode:
    def __init__(self, loss, param, seed=None, clear_gradients=True):
        self.calls = []
        self.modes = []
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

    def insert(self, func):
        assert func.autodiff_mode == AutodiffMode.NONE or func.autodiff_mode == AutodiffMode.FORWARD, "Inserted funcs should be forward or grad kernels (forward mode)."
        self.modes.append(func.autodiff_mode)
        func.autodiff_mode = AutodiffMode.FORWARD
        self.calls.append((func))

    def recover_kernels(self):
        assert self.entered, "Before recover the kernels, fwd mode manager must be entered."
        for calls, mode in zip(self.calls, self.modes):
            calls.autodiff_mode = mode
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


from typing import Dict

from taichi.lang.util import cook_dtype

import taichi as ti


class Checkpointer:
    def __init__(self, snode_tree=impl.root_snode_tree, verbose=True):
        self.prog = None  # A placeholder, will be replaced when calling `save` or `restore`
        # assert self.prog, "Checkpointer should be constructed after ti.init()."

        self.verbose = verbose

        self.root = None
        self.snode_tree = snode_tree  # Set to default snode tree by default
        self.backup_buffers = dict()
        self.free_backup_buffers = []

        self.forward_calls = []

        self.drop_spacing = 2
        self.drop_id = 2  # The dropping starts from the second buffer
        self.save_spacing = 1
        self.save_spacing_cnt = 1
        self.current_level_checkpoints_num = 0
        self.current_start_id = 1

        self.max_checkpointing_level = 4
        self.max_num_buffers_per_level = 10
        self.max_num_buffers = self.max_num_buffers_per_level * self.max_checkpointing_level
        # For stat use
        self.current_max_num_buffers = 0
        self.buffer_created = 0
        self.recompute_num = 0

    def print_info(self, info):
        print(f"[{__class__.__name__}]{info}")

    def register_variables(self, variables: Dict):
        self.root = ti.FieldsBuilder()
        for shape, var in variables.items():
            # TODO: handle more kinds of shapes
            if shape == ():
                self.root.dense((), ()).place(*var)
            elif isinstance(shape, int):
                self.root.dense(ti.i, shape).place(*var)
            elif len(shape) == 1:
                self.root.dense(ti.i, shape).place(*var)
            elif len(shape) == 2:
                self.root.dense(ti.ij, shape).place(*var)
            elif len(shape) == 3:
                self.root.dense(ti.ijk, shape).place(*var)
            else:
                raise NotImplementedError(f"Shape {shape} not supported.")
        self.snode_tree = self.root.finalize()

    def save(self, save_id, enforce=False):

        if not self.prog:
            self.prog = impl.get_runtime().prog
        assert self.prog, "Checkpointer should be called after ti.init()."
        assert self.snode_tree, "Please register variables which requires checkpointing."
        snode_tree_id = self.snode_tree.id

        if not enforce:
            # Every time starting a drop round, increase the save spacing by 1
            if self.save_spacing_cnt < self.save_spacing:
                self.save_spacing_cnt += 1
                return
            assert self.save_spacing_cnt <= self.save_spacing, "Save spacing cnt should not execeed save spacing."
            # Rest the save spacing cnt when a save operation is performed
            self.save_spacing_cnt = 1

        buffer_ptr = None
        if save_id not in self.backup_buffers:
            # Check whether requires drop
            if self.current_level_checkpoints_num >= self.max_num_buffers_per_level:
                if self.verbose:
                    self.print_info(
                        f"[SAVE] Current level limitation reached. Save id: {save_id}, Drop: {self.drop_id} backup buffers: {self.backup_buffers.keys()}"
                    )
                self.clear_checkpoint(self.drop_id)
                self.drop_id += self.drop_spacing
                if self.drop_id > save_id:
                    self.save_spacing *= 2
                    self.drop_id = self.current_start_id + self.drop_spacing
                    self.drop_spacing *= 2

            if len(self.free_backup_buffers) > 0:
                # Reuse free buffer ptr
                buffer_ptr = self.free_backup_buffers.pop()
            else:
                assert self.get_total_checkpoint_num(
                ) <= self.max_num_buffers, f"Run out of max number of buffers: {self.max_num_buffers}."
                buffer_ptr = self.prog.create_snode_tree_root_buffer_backup(
                    cook_dtype(int), snode_tree_id)
                self.buffer_created += 1
                if self.verbose:
                    self.print_info(
                        f"[SAVE] Buffer created: {self.buffer_created}")
        else:
            # Overwritten if the save id already exists.
            buffer_ptr = self.backup_buffers[save_id]

        self.prog.save_snode_tree_root_buffer(buffer_ptr, snode_tree_id)
        self.current_level_checkpoints_num += 1
        self.backup_buffers[save_id] = buffer_ptr
        if self.verbose:
            self.print_info(
                f"[SAVE] Checkpoint {save_id} saved from snode tree {snode_tree_id}"
            )

        num_buffers = self.get_total_checkpoint_num()
        if num_buffers > self.current_max_num_buffers:
            self.current_max_num_buffers = num_buffers

    @no_grad
    def restore(self, save_id):
        if not self.prog:
            self.prog = impl.get_runtime().prog
        assert self.prog, "Checkpointer should be called after ti.init()."
        assert self.snode_tree, "Please register variables which requires checkpointing."
        snode_tree_id = self.snode_tree.id

        # Enter into recompute routine if save id not found
        if not self.find_checkpoint(save_id):
            last_save_id = self.get_last_checkpoint_id()
            if self.verbose:
                self.print_info(
                    f"[RESTORE] Checkpoint {save_id} not found, recompute start from {last_save_id}."
                )
            self.reset_checkpointing_for_new_level(last_save_id)

            buffer_ptr = self.backup_buffers[last_save_id]
            self.prog.restore_snode_tree_root_buffer(buffer_ptr, snode_tree_id)

            for i in range(last_save_id, save_id + 1):
                # Because the save id starts from 1
                func, args = self.forward_calls[i - 1]
                func(*args)

                if i == save_id:
                    # The last checkpoint must be saved
                    self.save(i, enforce=True)
                else:
                    self.save(i)
                if self.verbose:
                    self.print_info(f"[RECOMPUTE] {i}, target {save_id}")

                # Record total recompute num
                self.recompute_num += 1

        buffer_ptr = self.backup_buffers[save_id]
        self.prog.restore_snode_tree_root_buffer(buffer_ptr, snode_tree_id)
        self.clear_checkpoint(save_id)

        if self.verbose:
            self.print_info(
                f"[RESTORE] Checkpoint {save_id} restored to snode tree {snode_tree_id}"
            )

    def find_checkpoint(self, save_id):
        return save_id in self.backup_buffers

    def get_last_checkpoint_id(self):
        return max(self.backup_buffers.keys())

    def get_total_checkpoint_num(self):
        return len(self.backup_buffers.keys())

    def reset(self, start_id=1):
        self.reset_checkpointing_for_new_level(start_id)
        self.recompute_num = 0

    def reset_checkpointing_for_new_level(self, start_id):
        self.drop_spacing = 2
        self.current_start_id = start_id
        self.drop_id = self.current_start_id + 1  # The dropping starts from the second buffer
        self.save_spacing = 1
        self.save_spacing_cnt = 1
        self.current_level_checkpoints_num = 0

    def clear_checkpoint(self, save_id):
        buffer = self.backup_buffers.pop(save_id, None)
        assert buffer, f"Buffer with save id {save_id} doesn't exist."
        self.free_backup_buffers.append(buffer)

    def clear(self):
        if self.verbose:
            self.print_info(
                f"[CLEAR] Total buffers created: {self.buffer_created}.")
            self.print_info(
                f"[CLEAR] Total recompute num: {self.recompute_num}.")
        save_ids = list(self.backup_buffers.keys())
        # print("Clearing checkpointer, remaining save ids: ", save_ids)
        # print("max buffer created ", self.current_max_num_buffers)
        for save_id in save_ids:
            self.clear_checkpoint(save_id)
        self.current_level_checkpoints_num = 0


class CheckpointerManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.forward_result_checpointer = Checkpointer(verbose=True)
        self.primal_checkpointer = Checkpointer(verbose=verbose)
        self.grad_checkpointer = Checkpointer(
            snode_tree=impl.root_grad_snode_tree, verbose=verbose)

    def register_variables(self, variables: Dict):
        self.primal_checkpointer.register_variables(variables)
        variables_grad = {}
        for shape, fields in variables.items():
            variables_grad[shape] = [f.grad for f in fields]
        self.grad_checkpointer.register_variables(variables_grad)

    def save_primal(self, save_id):
        self.primal_checkpointer.save(save_id)

    def save_forward_results(self):
        self.forward_result_checpointer.save(0, enforce=True)

    def reset(self):
        self.primal_checkpointer.reset()
        self.forward_result_checpointer.reset()

    def restore_primal(self, save_id):
        self.primal_checkpointer.restore(save_id)

    def restore_forward_results(self):
        self.forward_result_checpointer.restore(0)

    def get_forward_calls(self, calls):
        self.primal_checkpointer.forward_calls = calls

    def clear_primal_checkpoint(self, save_id):
        self.primal_checkpointer.clear_checkpoint(save_id)

    def clear_forward_results(self):
        self.forward_result_checpointer.clear()

    def clear(self):
        self.primal_checkpointer.clear()
        self.forward_result_checpointer.clear()


_checkpointer = CheckpointerManager(verbose=False)

__all__ = [
    'FwdMode', 'Tape', 'clear_all_gradients', 'grad_for', 'grad_replaced',
    'no_grad'
]
