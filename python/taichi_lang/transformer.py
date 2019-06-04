import ast

class ASTTransformer(ast.NodeTransformer):
  def __init__(self):
    super().__init__()
    self.local_scopes = []

  def push_scope(self):
    self.local_scopes.append([])

  def pop_scope(self):
    self.local_scopes = self.local_scopes[:-1]

  def current_scope(self):
    return self.local_scopes[-1]

  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    init = ast.Attribute(
      value=ast.Name(id='ti', ctx=ast.Load()), attr='expr_init',
      ctx=ast.Load())
    rhs = ast.Call(
      func=init,
      args=[node.value],
      keywords=[],
    )
    return ast.copy_location(ast.Assign(targets=node.targets, value=rhs), node)

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    return node

