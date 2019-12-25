import ast
from .util import to_taichi_type


class TaichiSyntaxError(Exception):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class ScopeGuard:

  def __init__(self, t, stmt_block=None):
    self.t = t
    self.stmt_block = stmt_block

  def __enter__(self):
    self.t.local_scopes.append([])

  def __exit__(self, exc_type, exc_val, exc_tb):
    local = self.t.local_scopes[-1]
    if self.stmt_block is not None:
      for var in reversed(local):
        stmt = ASTTransformer.parse_stmt('del var')
        stmt.targets[0].id = var
        self.stmt_block.append(stmt)
    self.t.local_scopes = self.t.local_scopes[:-1]


# Single-pass transform
class ASTTransformer(ast.NodeTransformer):

  def __init__(self,
               excluded_paremeters=(),
               is_kernel=True,
               func=None,
               arg_features=None):
    super().__init__()
    self.local_scopes = []
    self.excluded_parameters = excluded_paremeters
    self.is_kernel = is_kernel
    self.func = func
    self.arg_features = arg_features

  def variable_scope(self, *args):
    return ScopeGuard(self, *args)

  def current_scope(self):
    return self.local_scopes[-1]

  def var_declared(self, name):
    for s in self.local_scopes:
      if name in s:
        return True
    return False

  def is_creation(self, name):
    return not self.var_declared(name)

  def create_variable(self, name):
    assert name not in self.current_scope(
    ), "Recreating variables is not allowed"
    self.current_scope().append(name)

  def generic_visit(self, node, body_names=[]):
    assert isinstance(body_names, list)
    for field, old_value in ast.iter_fields(node):
      if field in body_names:
        list_stmt = old_value
        with self.variable_scope(list_stmt):
          for i, l in enumerate(list_stmt):
            list_stmt[i] = self.visit(l)

      elif isinstance(old_value, list):
        new_values = []
        for value in old_value:
          if isinstance(value, ast.AST):
            value = self.visit(value)
            if value is None:
              continue
            elif not isinstance(value, ast.AST):
              new_values.extend(value)
              continue
          new_values.append(value)
        old_value[:] = new_values
      elif isinstance(old_value, ast.AST):
        new_node = self.visit(old_value)
        if new_node is None:
          delattr(node, field)
        else:
          setattr(node, field, new_node)
    return node

  def visit_AugAssign(self, node):
    self.generic_visit(node)
    template = 'x.augassign(0, 0)'
    t = ast.parse(template).body[0]
    t.value.func.value = node.target
    t.value.func.value.ctx = ast.Load()
    t.value.args[0] = node.value
    t.value.args[1] = ast.Str(s=type(node.op).__name__, ctx=ast.Load())
    return ast.copy_location(t, node)

  @staticmethod
  def make_single_statement(stmts):
    template = 'if 1: pass'
    t = ast.parse(template).body[0]
    t.body = stmts
    return t

  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    self.generic_visit(node)

    if isinstance(node.targets[0], ast.Tuple):
      targets = node.targets[0].elts

      # Create
      stmts = []

      holder = self.parse_stmt('__tmp_tuple = 0')
      holder.value = node.value

      stmts.append(holder)

      def tuple_indexed(i):
        indexing = self.parse_stmt('__tmp_tuple[0]')
        indexing.value.slice.value = self.parse_expr("{}".format(i))
        return indexing.value

      for i, target in enumerate(targets):
        is_local = isinstance(target, ast.Name)
        if is_local and self.is_creation(target.id):
          var_name = target.id
          target.ctx = ast.Store()
          # Create
          init = ast.Attribute(
              value=ast.Name(id='ti', ctx=ast.Load()),
              attr='expr_init',
              ctx=ast.Load())
          rhs = ast.Call(
              func=init,
              args=[tuple_indexed(i)],
              keywords=[],
          )
          self.create_variable(var_name)
          stmts.append(ast.Assign(targets=[target], value=rhs))
        else:
          # Assign
          target.ctx = ast.Load()
          func = ast.Attribute(value=target, attr='assign', ctx=ast.Load())
          call = ast.Call(func=func, args=[tuple_indexed(i)], keywords=[])
          stmts.append(ast.Expr(value=call))

      for stmt in stmts:
        ast.copy_location(stmt, node)
      stmts.append(self.parse_stmt('del __tmp_tuple'))
      return self.make_single_statement(stmts)
    else:
      is_local = isinstance(node.targets[0], ast.Name)
      if is_local and self.is_creation(node.targets[0].id):
        var_name = node.targets[0].id
        # Create
        init = ast.Attribute(
            value=ast.Name(id='ti', ctx=ast.Load()),
            attr='expr_init',
            ctx=ast.Load())
        rhs = ast.Call(
            func=init,
            args=[node.value],
            keywords=[],
        )
        self.create_variable(var_name)
        return ast.copy_location(
            ast.Assign(targets=node.targets, value=rhs), node)
      else:
        # Assign
        node.targets[0].ctx = ast.Load()
        func = ast.Attribute(
            value=node.targets[0], attr='assign', ctx=ast.Load())
        call = ast.Call(func=func, args=[node.value], keywords=[])
        return ast.copy_location(ast.Expr(value=call), node)

  def visit_Try(self, node):
    raise TaichiSyntaxError("Keyword 'try' not supported in Taichi kernels")

  def visit_Import(self, node):
    raise TaichiSyntaxError("Keyword 'import' not supported in Taichi kernels")

  def visit_While(self, node):
    if node.orelse:
      raise TaichiSyntaxError(
        "'else' clause for 'while' not supported in Taichi kernels")

    template = ''' 
if 1:
  ti.core.begin_frontend_while(ti.Expr(1).ptr)
  __while_cond = 0
  if __while_cond:
    pass
  else:
    break
  ti.core.pop_scope()
'''
    cond = node.test
    t = ast.parse(template).body[0]
    t.body[1].value = cond
    t.body = t.body[:3] + node.body + t.body[3:]

    self.generic_visit(t, ['body'])
    return ast.copy_location(t, node)

  def visit_block(self, list_stmt):
    for i, l in enumerate(list_stmt):
      list_stmt[i] = self.visit(l)

  def visit_If(self, node):
    self.generic_visit(node, ['body', 'orelse'])

    is_static_if = isinstance(node.test, ast.Call) and isinstance(
        node.test.func, ast.Attribute)
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

  def check_loop_var(self, loop_var):
    if self.var_declared(loop_var):
      raise TaichiSyntaxError(
          "Variable '{}' is already declared in the outer scope and cannot be used as loop variable"
          .format(loop_var))

  def visit_For(self, node):
    if node.orelse:
      raise TaichiSyntaxError(
          "'else' clause for 'for' not supported in Taichi kernels")
    self.generic_visit(node, ['body'])
    decorated = isinstance(node.iter, ast.Call) and isinstance(
        node.iter.func, ast.Attribute)
    is_static_for = False
    is_grouped = False
    if decorated:
      attr = node.iter.func
      if attr.attr == 'static':
        is_static_for = True
      elif attr.attr == 'grouped':
        is_grouped = True
    is_range_for = isinstance(node.iter, ast.Call) and isinstance(
        node.iter.func, ast.Name) and node.iter.func.id == 'range'
    if is_static_for:
      return node
    elif is_range_for:
      loop_var = node.target.id
      self.check_loop_var(loop_var)
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
    else:  # Struct for
      if isinstance(node.target, ast.Name):
        elts = [node.target]
      else:
        elts = node.target.elts

      for loop_var in elts:
        self.check_loop_var(loop_var.id)

      var_decl = ''.join(
          '  {} = ti.Expr(ti.core.make_id_expr(""))\n'.format(ind.id)
          for ind in elts)
      vars = ', '.join(ind.id for ind in elts)
      if is_grouped:
        template = ''' 
if 1:
  ___loop_var = 0
  {} = ti.make_var_vector(size=___loop_var.loop_range().dim())
  ___expr_group = ti.make_expr_group({})
  ti.core.begin_frontend_struct_for(___expr_group, ___loop_var.loop_range().ptr)
  ti.core.end_frontend_range_for()
        '''.format(vars, vars)
        t = ast.parse(template).body[0]
        cut = 4
        t.body[0].value = node.iter
        t.body = t.body[:cut] + node.body + t.body[cut:]
      else:
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
      for loop_var in reversed(elts):
        t.body.append(self.parse_stmt('del {}'.format(loop_var.id)))
      return ast.copy_location(t, node)

  @staticmethod
  def parse_stmt(stmt):
    return ast.parse(stmt).body[0]

  @staticmethod
  def parse_expr(expr):
    return ast.parse(expr).body[0].value

  @staticmethod
  def func_call(name, *args):
    return ast.Call(
        func=ASTTransformer.parse_expr(name).value,
        args=list(args),
        keywords=[])

  def visit_Subscript(self, node):
    self.generic_visit(node)

    value = node.value
    indices = node.slice
    if isinstance(indices.value, ast.Tuple):
      indices = indices.value.elts
    else:
      indices = [indices.value]

    call = ast.Call(
        func=self.parse_expr('ti.subscript'),
        args=[value] + indices,
        keywords=[])
    return ast.copy_location(call, node)

  def visit_IfExp(self, node):
    raise TaichiSyntaxError(
        'Ternary operator ("a if cond else b") is not yet supported in Taichi kernels. Please walk around by changing loop conditions.'
    )

  def visit_Break(self, node):
    return self.parse_stmt('ti.core.insert_break_stmt()')

  def visit_Continue(self, node):
    raise TaichiSyntaxError(
        '"continue" is not yet supported in Taichi kernels. Please walk around by changing loop conditions.'
    )

  def visit_Call(self, node):
    self.generic_visit(node)
    if isinstance(node.func, ast.Name):
      func_name = node.func.id
      if func_name == 'print':
        node.func = self.parse_expr('ti.ti_print')
      elif func_name == 'min':
        node.func = self.parse_expr('ti.ti_min')
      elif func_name == 'max':
        node.func = self.parse_expr('ti.ti_max')
      elif func_name == 'int':
        node.func = self.parse_expr('ti.ti_int')
      elif func_name == 'float':
        node.func = self.parse_expr('ti.ti_float')
      else:
        pass
    return node

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
    args = node.args
    assert args.vararg is None
    assert args.kwonlyargs == []
    assert args.kw_defaults == []
    assert args.kwarg is None
    import taichi as ti
    if self.is_kernel:
      # Transform as kernel
      arg_decls = []
      for i, arg in enumerate(args.args):
        if isinstance(self.func.arguments[i], ti.template):
          continue
        import taichi as ti
        if isinstance(self.func.arguments[i], ti.ext_arr):
          arg_init = self.parse_stmt('x = ti.decl_ext_arr_arg(0, 0)')
          arg_init.targets[0].id = arg.arg
          self.create_variable(arg.arg)
          array_dt = self.arg_features[i][0]
          array_dim = self.arg_features[i][1]
          import numpy as np
          array_dt = to_taichi_type(array_dt)
          if array_dt == ti.f32:
            dt = self.parse_expr('ti.f32')
          elif array_dt == ti.f64:
            dt = self.parse_expr('ti.f64')
          elif array_dt == ti.i32:
            dt = self.parse_expr('ti.i32')
          elif array_dt == ti.i64:
            dt = self.parse_expr('ti.i64')
          else:
            assert False
          arg_init.value.args[0] = dt
          arg_init.value.args[1] = self.parse_expr("{}".format(array_dim))
          arg_decls.append(arg_init)
        else:
          arg_init = self.parse_stmt('x = ti.decl_scalar_arg(0)')
          arg_init.targets[0].id = arg.arg
          dt = arg.annotation
          arg_init.value.args[0] = dt
          arg_decls.append(arg_init)
      # remove original args
      node.args.args = []
    else:
      # Transform as func (all parameters passed by value)
      arg_decls = []
      for i, arg in enumerate(args.args):
        arg_init = self.parse_stmt('x = ti.expr_init(0)')
        arg_init.targets[0].id = arg.arg
        self.create_variable(arg.arg)
        arg_init.value.args[0] = self.parse_expr(arg.arg + '_by_value__')
        args.args[i].arg += '_by_value__'
        arg_decls.append(arg_init)
    with self.variable_scope():
      self.generic_visit(node)
    node.body = arg_decls + node.body
    return node

  def visit_UnaryOp(self, node):
    self.generic_visit(node)
    if isinstance(node.op, ast.Not):
      # Python does not support overloading logical and & or
      new_node = self.parse_expr('ti.logical_not(0)')
      new_node.args[0] = node.operand
      node = new_node
    return node
  
  def visit_Compare(self, node):
    self.generic_visit(node)
    ret = None
    comparators = [node.left] + node.comparators
    for i in range(len(node.comparators)):
      new_cmp = ast.Compare(left=comparators[i], ops=[node.ops[i]], comparators=[comparators[i + 1]])
      ast.copy_location(new_cmp, node)
      if ret is None:
        ret = new_cmp
      else:
        ret = ast.BoolOp(op=ast.And(), values=[ret, new_cmp])
        ret = self.visit_BoolOp(ret)
        ast.copy_location(ret, node)
        
    self.generic_visit(ret)
    # import astpretty
    # astpretty.pprint(ret)
    return ret
    

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
  
  def visit_Assert(self, node):
    import astor
    msg = astor.to_source(node.test)
    self.generic_visit(node)
    new_node = self.parse_stmt('ti.core.create_assert_stmt(ti.Expr(0).ptr, 0)')
    new_node.value.args[0].value.args[0] = node.test
    new_node.value.args[1] = self.parse_expr("'{}'".format(msg.strip()))
    return new_node
