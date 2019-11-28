import taichi as ti

# A set of helper template functions

@ti.kernel
def clear_matrix(mat: ti.template()):
  # TODO: ti.index_group
  pass

@ti.kernel
def tensor_to_numpy(tensor: ti.template(), arr: ti.ext_arr()):
  for I in ti.grouped(tensor):
    arr[I] = tensor[I]

@ti.kernel
def numpy_to_tensor(arr: ti.ext_arr(), tensor: ti.template()):
  for I in ti.grouped(tensor):
    tensor[I] = arr[I]

