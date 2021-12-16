from taichi.lang.ast.ast_transformer import ASTTransformer
from taichi.lang.ast.ast_transformer_utils import (ASTTransformerContext,
                                                   print_ast)


def transform_tree(tree, ctx: ASTTransformerContext):
    print_ast(tree, 'Initial AST')
    ASTTransformer()(ctx, tree)
    print_ast(tree, 'Preprocessed')
    return ctx.return_data
