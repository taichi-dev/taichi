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

