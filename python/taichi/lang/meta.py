import taichi as ti

# A set of helper (meta)functions

@ti.kernel
def fill_tensor(tensor: ti.template(), val: ti.template()):
  for I in ti.grouped(tensor):
    tensor[I] = val

@ti.kernel
def tensor_to_ext_arr(tensor: ti.template(), arr: ti.ext_arr()):
  for I in ti.grouped(tensor):
    arr[I] = tensor[I]

@ti.kernel
def ext_arr_to_tensor(arr: ti.ext_arr(), tensor: ti.template()):
  for I in ti.grouped(tensor):
    tensor[I] = arr[I]

@ti.kernel
def matrix_to_ext_arr(mat: ti.template(), arr: ti.ext_arr(), as_vector: ti.template()):
  for I in ti.grouped(mat):
    for p in ti.static(range(mat.n)):
      for q in ti.static(range(mat.m)):
        if ti.static(as_vector):
          arr[I, p] = mat[I][p]
        else:
          arr[I, p, q] = mat[I][p, q]
          
@ti.kernel
def ext_arr_to_matrix(arr: ti.ext_arr(), mat: ti.template(), as_vector: ti.template()):
  for I in ti.grouped(mat):
    for p in ti.static(range(mat.n)):
      for q in ti.static(range(mat.m)):
        if ti.static(as_vector):
          mat[I][p] = arr[I, p]
        else:
          mat[I][p, q] = arr[I, p, q]

@ti.kernel
def clear_gradients(vars: ti.template()):
  for I in ti.grouped(ti.Expr(vars[0])):
    for s in ti.static(vars):
      ti.Expr(s)[I] = 0
