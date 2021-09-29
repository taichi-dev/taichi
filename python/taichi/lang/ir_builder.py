import ast
import warnings

from taichi.lang.ast_builder_utils import *
from taichi.lang.ast_resolver import ASTResolver
from taichi.lang.exception import TaichiSyntaxError

import taichi as ti


class IRBuilder(Builder):
    @staticmethod
    def build_Name(ctx, node):
        node.ptr = ctx.get_var_by_name(node.id)
        return node

    @staticmethod
    def build_Assign(ctx, node):
        assert len(node.targets) == 1
        assert isinstance(node.targets[0], ast.Name)
        name = node.targets[0].id
        is_creation = ctx.is_creation(name)
        node.value = build_ir(ctx, node.value)
        if is_creation:
            ctx.create_variable(name, ti.expr_init(node.value.ptr))
        else:
            var = ctx.get_var_by_name(name)
            var.assign(node.value.ptr)

    @staticmethod
    def build_Constant(ctx, node):
        node.ptr = node.value
        return node

    @staticmethod
    def build_FunctionDef(ctx, node):
        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        arg_decls = []

        def transform_as_kernel():
            # Treat return type
            if node.returns is not None:
                node.returns = build_ir(ctx, node.returns)
                ti.lang.kernel_arguments.decl_scalar_ret(node.returns.ptr)
                ctx.returns = node.returns.ptr

            for i, arg in enumerate(args.args):
                # Directly pass in template arguments,
                # such as class instances ("self"), fields, SNodes, etc.
                # if isinstance(ctx.func.argument_annotations[i], ti.template):
                #     continue
                # if isinstance(ctx.func.argument_annotations[i],
                #               ti.sparse_matrix_builder):
                #     arg_init = parse_stmt(
                #         'x = ti.lang.kernel_arguments.decl_sparse_matrix()')
                #     arg_init.targets[0].id = arg.arg
                #     ctx.create_variable(arg.arg)
                #     arg_decls.append(arg_init)
                # elif isinstance(ctx.func.argument_annotations[i], ti.any_arr):
                #     arg_init = parse_stmt(
                #         'x = ti.lang.kernel_arguments.decl_any_arr_arg(0, 0, 0, 0)'
                #     )
                #     arg_init.targets[0].id = arg.arg
                #     ctx.create_variable(arg.arg)
                #     array_dt = ctx.arg_features[i][0]
                #     array_dim = ctx.arg_features[i][1]
                #     array_element_shape = ctx.arg_features[i][2]
                #     array_layout = ctx.arg_features[i][3]
                #     array_dt = to_taichi_type(array_dt)
                #     dt_expr = 'ti.' + ti.core.data_type_name(array_dt)
                #     dt = parse_expr(dt_expr)
                #     arg_init.value.args[0] = dt
                #     arg_init.value.args[1] = parse_expr("{}".format(array_dim))
                #     arg_init.value.args[2] = parse_expr(
                #         "{}".format(array_element_shape))
                #     arg_init.value.args[3] = parse_expr(
                #         "ti.{}".format(array_layout))
                #     arg_decls.append(arg_init)
                if False:
                    pass
                else:
                    arg.annotation = build_ir(ctx, arg.annotation)
                    ctx.create_variable(arg.arg, ti.lang.kernel_arguments.decl_scalar_arg(arg.annotation.ptr))
            # remove original args
            node.args.args = []

        if ctx.is_kernel:  # ti.kernel
            for decorator in node.decorator_list:
                if ASTResolver.resolve_to(decorator, ti.func, globals()):
                    raise TaichiSyntaxError(
                        "Function definition not allowed in 'ti.kernel'.")
            transform_as_kernel()
        #
        # else:  # ti.func
        #     for decorator in node.decorator_list:
        #         if ASTResolver.resolve_to(decorator, ti.func, globals()):
        #             raise TaichiSyntaxError(
        #                 "Function definition not allowed in 'ti.func'.")
        #     if impl.get_runtime().experimental_real_function:
        #         transform_as_kernel()
        #     else:
        #         # Transform as force-inlined func
        #         arg_decls = []
        #         for i, arg in enumerate(args.args):
        #             # Remove annotations because they are not used.
        #             args.args[i].annotation = None
        #             # Template arguments are passed by reference.
        #             if isinstance(ctx.func.argument_annotations[i],
        #                           ti.template):
        #                 ctx.create_variable(ctx.func.argument_names[i])
        #                 continue
        #             # Create a copy for non-template arguments,
        #             # so that they are passed by value.
        #             arg_init = parse_stmt('x = ti.expr_init_func(0)')
        #             arg_init.targets[0].id = arg.arg
        #             ctx.create_variable(arg.arg)
        #             arg_init.value.args[0] = parse_expr(arg.arg +
        #                                                 '_by_value__')
        #             args.args[i].arg += '_by_value__'
        #             arg_decls.append(arg_init)

        with ctx.variable_scope():
            build_irs(ctx, node.body)

        return node

    @staticmethod
    def build_Return(ctx, node):
        node.value = build_ir(ctx, node.value)
        if ctx.is_kernel or impl.get_runtime().experimental_real_function:
            # TODO: check if it's at the end of a kernel, throw TaichiSyntaxError if not
            if node.value is not None:
                if ctx.returns is None:
                    raise TaichiSyntaxError(
                        f'A {"kernel" if ctx.is_kernel else "function"} '
                        'with a return value must be annotated '
                        'with a return type, e.g. def func() -> ti.f32')
                ti.core.create_kernel_return(ti.cast(ti.Expr(node.value.ptr), ctx.returns).ptr)
                # For args[0], it is an ast.Attribute, because it loads the
                # attribute, |ptr|, of the expression |ret_expr|. Therefore we
                # only need to replace the object part, i.e. args[0].value
        return node

    @staticmethod
    def build_Module(ctx, node):
        with ctx.variable_scope():
            # Do NOT use |build_stmts| which inserts 'del' statements to the
            # end and deletes parameters passed into the module
            node.body = [build_ir(ctx, stmt) for stmt in list(node.body)]
        return node

    @staticmethod
    def build_Attribute(ctx, node):
        node.value = build_ir(ctx, node.value)
        node.ptr = getattr(node.value.ptr, node.attr)
        return node

    @staticmethod
    def build_BinOp(ctx, node):
        node.left = build_ir(ctx, node.left)
        node.right = build_ir(ctx, node.right)
        op = {
            ast.Add: lambda l, r: l + r,
            ast.Sub: lambda l, r: l - r,
            ast.Mult: lambda l, r: l * r,
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
        }.get(type(node.op))
        node.ptr = op(node.left.ptr, node.right.ptr)
        return node



build_ir = IRBuilder()


def build_irs(ctx, stmts):
    result = []
    with ctx.variable_scope(result):
        for stmt in list(stmts):
            result.append(build_ir(ctx, stmt))
    return result
