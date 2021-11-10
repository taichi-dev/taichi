import sys
import traceback

import numpy as np
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.util import is_taichi_class, python_scope

import taichi as ti


# Scalar, basic data type
class Expr(TaichiOperations):
    """A Python-side Expr wrapper, whose member variable `ptr` is an instance of C++ Expr class. A C++ Expr object contains member variable `expr` which holds an instance of C++ Expression class."""
    def __init__(self, *args, tb=None):
        _taichi_skip_traceback = 1
        self.tb = tb
        if len(args) == 1:
            if isinstance(args[0], _ti_core.Expr):
                self.ptr = args[0]
            elif isinstance(args[0], Expr):
                self.ptr = args[0].ptr
                self.tb = args[0].tb
            elif is_taichi_class(args[0]):
                raise ValueError('cannot initialize scalar expression from '
                                 f'taichi class: {type(args[0])}')
            else:
                # assume to be constant
                arg = args[0]
                try:
                    if isinstance(arg, np.ndarray):
                        arg = arg.dtype(arg)
                except:
                    pass
                self.ptr = impl.make_constant_expr(arg).ptr
        else:
            assert False
        if self.tb:
            self.ptr.set_tb(self.tb)
        try:
            self.ptr.type_check()
        except RuntimeError as e:
            if str(e).startswith('TypeError: '):
                s = traceback.extract_stack()
                for i, l in enumerate(s):
                    if 'taichi_ast_generator' in l:
                        s = s[i + 1:]
                        break
                print('[Taichi] Compilation failed', file=sys.stderr)
                print(traceback.format_list(s[:1])[0], end='', file=sys.stderr)
                print(f'TaichiTypeError: {str(e)[11:]}', file=sys.stderr)
                sys.exit(1)
            raise e

    def __hash__(self):
        return self.ptr.get_raw_address()

    def __str__(self):
        return '<ti.Expr>'

    def __repr__(self):
        return '<ti.Expr>'


def make_var_list(size):
    exprs = []
    for _ in range(size):
        exprs.append(_ti_core.make_id_expr(''))
    return exprs


def make_expr_group(*exprs):
    if len(exprs) == 1:
        if isinstance(exprs[0], (list, tuple)):
            exprs = exprs[0]
        elif isinstance(exprs[0], ti.Matrix):
            mat = exprs[0]
            assert mat.m == 1
            exprs = mat.entries
    expr_group = _ti_core.ExprGroup()
    for i in exprs:
        if isinstance(i, ti.Matrix):
            assert i.local_tensor_proxy is not None
            expr_group.push_back(i.local_tensor_proxy)
        else:
            expr_group.push_back(Expr(i).ptr)
    return expr_group
