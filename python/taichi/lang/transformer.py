import ast


class ScopeGuard:
  def __init__(self, t):
    self.t = t

  def __enter__(self):
    self.t.local_scopes.append(set())

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.t.local_scopes = self.t.local_scopes[:-1]


# Single-pass transform
class ASTTransformer(ast.NodeTransformer):
  def __init__(self, excluded_paremeters=(), transform_args=True):
    super().__init__()
    self.local_scopes = []
    self.excluded_parameters = excluded_paremeters
    self.transform_args = transform_args

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

  def visit_AugAssign(self, node):
    self.generic_visit(node)
    template = 'x.augassign(0, 0)'
    t = ast.parse(template).body[0]
    t.value.func.value = node.target
    t.value.func.value.ctx = ast.Load()
    t.value.args[0] = node.value
    t.value.args[1] = ast.Str(s=type(node.op).__name__, ctx=ast.Load())
    return ast.copy_location(t, node)

  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    self.generic_visit(node)
    is_local = isinstance(node.targets[0], ast.Name)

    if isinstance(node.targets[0], ast.Tuple):
      # Create
      init = ast.Attribute(
        value=ast.Name(id='ti', ctx=ast.Load()), attr='expr_init',
        ctx=ast.Load())
      rhs = ast.Call(
        func=init,
        args=[node.value],
        keywords=[],
      )
      for var in node.targets[0].elts:
        self.create_variable(var.id)
      return ast.copy_location(ast.Assign(targets=node.targets, value=rhs),
                               node)

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

  def visit_While(self, node):
    with self.variable_scope():
      self.generic_visit(node)

    template = ''' 
if 1:
  __cond = 0
  ti.core.begin_frontend_while(ti.Expr(__cond).ptr)
  ti.core.pop_scope()
'''
    t = ast.parse(template).body[0]
    cond = node.test
    t.body[0].value = cond
    t.body = t.body[:2] + node.body + t.body[2:]
    return ast.copy_location(t, node)

  def visit_block(self, list_stmt):
    for i, l in enumerate(list_stmt):
      list_stmt[i] = self.visit(l)

  def visit_If(self, node):
    with self.variable_scope():
      node.test = self.visit(node.test)
      with self.variable_scope():
        self.visit_block(node.body)
      with self.variable_scope():
        self.visit_block(node.orelse)

    is_static_if = isinstance(node.test,
                               ast.Call) and isinstance(node.test.func,
                                                        ast.Attribute)
    if is_static_if:
      attr = node.test.func
      if attr.attr == 'static':
        is_static_if = True
      else:
        is_static_if = False

    if is_static_if:
      # Do nothing
      return node

    template = ''' 
if 1:
  __cond = 0
  ti.core.begin_frontend_if(ti.Expr(__cond).ptr)
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
  ___begin = ti.cast(___begin, ti.i32)
  ___end = ti.cast(___end, ti.i32)
  ti.core.begin_frontend_range_for({}.ptr, ___begin.ptr, ___end.ptr)
  ti.core.end_frontend_range_for()
      '''.format(loop_var, loop_var)
      t = ast.parse(template).body[0]

      assert len(node.iter.args) in [1, 2]
      if len(node.iter.args) == 2:
        bgn = node.iter.args[0]
        end = node.iter.args[1]
      else:
        bgn = self.make_constant(value=0)
        end = node.iter.args[0]

      t.body[1].value.args[0] = bgn
      t.body[2].value.args[0] = end
      t.body = t.body[:6] + node.body + t.body[6:]
      t.body.append(self.parse_stmt('del {}'.format(loop_var)))
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
      template = ''' 
if 1:
{}
  ___loop_var = 0
  ___expr_group = ti.make_expr_group({})
  ti.core.begin_frontend_struct_for(___expr_group, ___loop_var.loop_range().ptr)
  ti.core.end_frontend_range_for()
      '''.format(var_decl, vars)
      t = ast.parse(template).body[0]
      cut = len(elts) + 3
      t.body[cut - 3].value = node.iter
      t.body = t.body[:cut] + node.body + t.body[cut:]
      return ast.copy_location(t, node)

  @staticmethod
  def parse_stmt(stmt):
    return ast.parse(stmt).body[0]

  @staticmethod
  def parse_expr(expr):
    return ast.parse(expr).body[0].value

  @staticmethod
  def func_call(name, *args):
    return ast.Call(func=ASTTransformer.parse_expr(name).value, args=list(args),
                    keywords=[])

  def visit_Subscript(self, node):
    self.generic_visit(node)

    value = node.value
    indices = node.slice
    if isinstance(indices.value, ast.Tuple):
      indices = indices.value.elts
    else:
      indices = [indices.value]

    call = ast.Call(func=self.parse_expr('ti.subscript'),
                    args=[value] + indices, keywords=[])
    return ast.copy_location(call, node)

  def visit_Module(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    return node

  def visit_Global(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    for name in node.names:
      self.create_variable(name)
    return node

  @staticmethod
  def make_constant(value):
    # Do not use ast.Constant which does not exist in python3.5
    node = ASTTransformer.parse_expr('0')
    node.value = value
    return node

  def visit_FunctionDef(self, node):
    with self.variable_scope():
      self.generic_visit(node)
    args = node.args
    assert args.vararg is None
    assert args.kwonlyargs == []
    assert args.kw_defaults == []
    assert args.kwarg is None
    if self.transform_args:
      arg_decls = []
      for i, arg in enumerate(args.args):
        if i in self.excluded_parameters:
          continue # skip template parameters
        arg_init = self.parse_stmt('x = ti.decl_arg(0)')
        arg_init.targets[0].id = arg.arg
        arg_init.value.args[0] = arg.annotation
        arg_decls.append(arg_init)
      node.body = arg_decls + node.body
      # remove original args
      node.args.args = []
    return node

  def visit_UnaryOp(self, node):
    self.generic_visit(node)
    if isinstance(node.op, ast.Not):
      # Python does not support overloading logical and & or
      new_node = self.parse_expr('ti.logical_not(0)')
      new_node.args[0] = node.operand
      node = new_node
    return node

  def visit_BoolOp(self, node):
    self.generic_visit(node)

    def make_node(a, b, token):
      new_node = self.parse_expr('ti.logical_{}(0, 0)'.format(token))
      new_node.args[0] = a
      new_node.args[1] = b
      return new_node

    token = ''
    if isinstance(node.op, ast.And):
      token = 'and'
    elif isinstance(node.op, ast.Or):
      token = 'or'
    else:
      print(node.op)
      print("BoolOp above not implemented")
      exit(0)

    new_node = node.values[0]
    for i in range(1, len(node.values)):
      new_node = make_node(new_node, node.values[i], token)

    return new_node
