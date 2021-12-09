from taichi.lang.ast.ast_transformer import ASTTransformer
from taichi.lang.ast.ast_transformer_check import ASTTransformerChecks
from taichi.lang.ast.ast_transformer_utils import ASTTransformerContext


def transform_tree(tree, ctx: ASTTransformerContext):
    ASTTransformer()(ctx, tree)
    ASTTransformerChecks(func=ctx.func,
                         global_vars=ctx.global_vars).visit(tree)
    return ctx.return_data
