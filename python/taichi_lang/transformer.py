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
      return ast.copy_location(ast.Assign(targets=node.targets, value=rhs),
                               node)
    else:
      # Assign
      node.targets[0].ctx = ast.Load()
      func = ast.Attribute(value=node.targets[0], attr='assign', ctx=ast.Load())
      call = ast.Call(func=func, args=[node.value], keywords=[])
      return ast.copy_location(ast.Expr(value=call), node)

  def visit_If(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    template = ''' 
if 1:
  __cond = 0
  ti.core.begin_frontend_if(__cond.ptr)
  ti.core.begin_frontend_if_true()
  ti.core.pop_scope()
  ti.core.begin_frontend_if_false()
  ti.core.pop_scope()
'''
    t = ast.parse(template).body[0]
    cond = node.test
    t.body[0].value = cond
    t.body = t.body[:5] + node.orelse + t.body[5:]
    t.body = t.body[:3] + node.body + t.body[3:]
    return ast.copy_location(t, node)

  def visit_For(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    is_static_for = isinstance(node.iter,
                              ast.Call) and isinstance(node.iter.func,
                                                       ast.Attribute)
    if is_static_for:
      attr = node.iter.func
      if attr.attr == 'static':
        is_static_for = True
      else:
        is_static_for = False
    is_range_for = isinstance(node.iter,
                              ast.Call) and isinstance(node.iter.func,
                                                       ast.Name) and node.iter.func.id == 'range'
    if is_static_for:
      return node
    elif is_range_for:
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
      # print(astor.to_source(t))
      return ast.copy_location(t, node)
    else:
      if isinstance(node.target, ast.Name):
        elts = [node.target]
      else:
        elts = node.target.elts

      var_decl = ''.join(
        '  {} = ti.Expr(ti.core.make_id_expr(""))\n'.format(ind.id) for ind in
        elts)
      vars = ', '.join(ind.id for ind in elts)
      # print(var_decl)
      template = ''' 
if 1:
{}
  ___loop_var = 0
  ___expr_group = ti.make_expr_group({})
  ti.core.begin_frontend_struct_for(___expr_group, ___loop_var.ptr)
  ti.core.end_frontend_range_for()
      '''.format(var_decl, vars)
      t = ast.parse(template).body[0]
      cut = len(elts) + 3
      t.body[cut - 3].value = node.iter
      t.body = t.body[:cut] + node.body + t.body[cut:]
      return ast.copy_location(t, node)

  @staticmethod
  def parse_expr(expr):
    return ast.parse(expr).body[0]

  @staticmethod
  def func_call(name, *args):
    return ast.Call(func=ASTTransformer.parse_expr(name).value, args=list(args),
                    keywords=[])

  def visit_Subscript(self, node):
    value = node.value
    indices = node.slice
    if isinstance(indices.value, ast.Tuple):
      indices = indices.value.elts
    else:
      indices = [indices.value]

    self.generic_visit(node)

    call = ast.Call(func=self.parse_expr('ti.subscript').value,
                    args=[value] + indices, keywords=[])
    return ast.copy_location(call, node)

  def visit_Module(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    return node
