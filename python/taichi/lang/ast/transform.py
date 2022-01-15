from taichi.lang.ast.ast_transformer import ASTTransformer
from taichi.lang.ast.ast_transformer_utils import ASTTransformerContext


def transform_tree(ast_builder, tree, ctx: ASTTransformerContext):
    assert ast_builder is not None
    ASTTransformer()(ast_builder, ctx, tree)
    return ctx.return_data
