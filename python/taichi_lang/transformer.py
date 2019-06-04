import ast

class ASTTransformer(ast.NodeTransformer):
  def __init__(self):
    super().__init__()
    self.local_scopes = []

  def push_scope(self):
    self.local_scopes.append(set())

  def pop_scope(self):
    self.local_scopes = self.local_scopes[:-1]

  def current_scope(self):
    return self.local_scopes[-1]

  def is_creation(self, name):
    return name not in self.current_scope()

  def create_variable(self, name):
    self.current_scope().add(name)

  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    assert isinstance(node.targets[0], ast.Name)
    var_name = node.targets[0].id
    if self.is_creation(var_name):
      # Create
      init = ast.Attribute(
        value=ast.Name(id='ti', ctx=ast.Load()), attr='expr_init',
        ctx=ast.Load())
      rhs = ast.Call(
        func=init,
        args=[node.value],
        keywords=[],
      )
      self.create_variable(var_name)
      return ast.copy_location(ast.Assign(targets=node.targets, value=rhs), node)
    else:
      # Assign
      func = ast.Attribute(value=ast.Name(id=var_name, ctx=ast.Load()), attr='assign', ctx=ast.Load())
      call = ast.Call(func=func, args=[node.value], keywords=[])
      return ast.copy_location(ast.Expr(value=call), node)

  def visit_Module(self, node):
    self.push_scope()
    self.generic_visit(node)
    self.pop_scope()
    return node

  def visit_FunctionDef(self, node):
    self.push_scope()
    self.generic_visit(node)
    self.pop_scope()
    return node

