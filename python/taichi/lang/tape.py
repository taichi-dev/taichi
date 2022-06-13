# from taichi.lang.impl import get_runtime
#
# class TapeImpl:
#     def __init__(self, loss=None, clear_gradients=True):
#         self.calls = []
#         self.entered = False
#         self.gradient_evaluated = False
#         self.runtime = get_runtime()
#         self.eval_on_exit = loss is not None
#         self.loss = loss
#
#     def __enter__(self):
#         self.runtime.target_tape = self
#         assert not self.entered, "Tape can be entered only once."
#         self.entered = True
#
#         get_runtime().materialize()
#         if len(self.loss.shape) != 0:
#             raise RuntimeError(
#                 'The loss of `Tape` must be a 0-D field, i.e. scalar')
#         if not self.loss.snode.ptr.has_adjoint():
#             raise RuntimeError(
#                 'Gradients of loss are not allocated, please use ti.field(..., needs_grad=True)'
#                 ' for all fields that are required by autodiff.')
#         if clear_gradients:
#             clear_all_gradients()
#
#         from taichi._kernels import clear_loss  # pylint: disable=C0415
#         clear_loss(loss)
#
#     def __exit__(self, _type, value, tb):
#         # print('# kernel calls', len(self.calls))
#         self.runtime.target_tape = None
#         if self.eval_on_exit:
#             self.grad()
#
#     def insert(self, func, args):
#         self.calls.append((func, args))
#
#     def grad(self):
#         assert self.entered, "Before evaluating gradients tape must be entered."
#         assert not self.gradient_evaluated, "Gradients of grad can be evaluated only once."
#         for func, args in reversed(self.calls):
#             func.grad(*args)
#         self.gradient_evaluated = True
