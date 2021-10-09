from taichi.lang.kernel_impl import kernel
from taichi.type.annotations import ext_arr, template


@kernel
def from_torch_template(expr: template(), torch_tensor: ext_arr()):
    for i in expr:
        expr[i] = torch_tensor[i]


@kernel
def to_torch_template(expr: template(), torch_tensor: ext_arr()):
    for i in expr:
        torch_tensor[i] = expr[i]


def from_torch(expr, torch_tensor):
    if not expr.from_torch_:
        expr.from_torch_ = lambda x: from_torch_template(expr, x.contiguous())
    expr.from_torch_(torch_tensor)


def to_torch(expr, torch_tensor):
    if not expr.to_torch_:
        expr.to_torch_ = lambda x: to_torch_template(expr, x.contiguous())
    expr.to_torch_(torch_tensor)
