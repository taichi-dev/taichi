import ast
import astpretty
import astor

class ScopeGuard:
  def __init__(self, t):
    self.t = t

  def __enter__(self):
    self.t.local_scopes.append(set())

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.t.local_scopes = self.t.local_scopes[:-1]

# Single-pass transform
class ASTTransformer(ast.NodeTransformer):
  def __init__(self):
    super().__init__()
    self.local_scopes = []

  def variable_scope(self):
    return ScopeGuard(self)

  def current_scope(self):
    return self.local_scopes[-1]

  def is_creation(self, name):
    for s in self.local_scopes:
      if name in s:
        return False
    return True

  def create_variable(self, name):
    self.current_scope().add(name)

  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    self.generic_visit(node)
    is_local = isinstance(node.targets[0], ast.Name)
    if is_local and self.is_creation(node.targets[0].id):
      var_name = node.targets[0].id
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
      node.targets[0].ctx = ast.Load()
      func = ast.Attribute(value=node.targets[0], attr='assign', ctx=ast.Load())
      call = ast.Call(func=func, args=[node.value], keywords=[])
      return ast.copy_location(ast.Expr(value=call), node)

  def visit_For(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    loop_var = node.target.id
    template = ''' 
if 1:
  {} = ti.Expr(ti.core.make_id_expr(''))
  ___begin = ti.Expr(0) 
  ___end = ti.Expr(0)
  ti.core.begin_frontend_range_for({}.ptr, ___begin.ptr, ___end.ptr)
  ti.core.end_frontend_range_for()
    '''.format(loop_var, loop_var)
    t = ast.parse(template).body[0]
    bgn = node.iter.args[0]
    end = node.iter.args[1]
    t.body[1].value.args[0] = bgn
    t.body[2].value.args[0] = end
    t.body = t.body[:4] + node.body + t.body[4:]
    print(astor.to_source(t))
    return ast.copy_location(t, node)

  @staticmethod
  def parse_expr(expr):
    return ast.parse(expr).body[0]

  @staticmethod
  def func_call(name, *args):
    return ast.Call(func=ASTTransformer.parse_expr(name).value, args=list(args), keywords=[])

  def visit_Subscript(self, node):
    value = node.value
    indices = node.slice
    if isinstance(indices, ast.Tuple):
      indices = indices.value.elts
    else:
      indices = [indices.value]

    self.generic_visit(node)

    call = ast.Call(func=self.parse_expr('ti.subscript').value, args=[value] + indices, keywords=[])
    return ast.copy_location(call, node)

  def visit_Module(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    return node

