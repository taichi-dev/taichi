#pragma once
#include <taichi/program/callable.h>
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/program/matrix.h"

namespace taichi::lang {

struct RuntimeContext;

class LaunchContextBuilder {
 public:
  enum class DevAllocType : int8_t {
    kNone = 0,
    kNdarray = 1,
    kTexture = 2,
    kRWTexture = 3
  };

  explicit LaunchContextBuilder(CallableBase *kernel);

  LaunchContextBuilder(LaunchContextBuilder &&) = default;
  LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
  LaunchContextBuilder(const LaunchContextBuilder &) = delete;
  LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

  void set_arg_float(int arg_id, float64 d);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(int arg_id, int64 d);
  void set_arg_uint(int arg_id, uint64 d);

  void set_array_runtime_size(int i, uint64 size);

  void set_array_device_allocation_type(int i, DevAllocType usage);

  template <typename T>
  void set_arg(int i, T v);

  // The following two functions can be used to set struct args and primitive
  // args. The first element of `arg_indices` is the index of the argument. The
  // rest of the elements are the index of the field in each depth of the nested
  // struct.

  template <typename T>
  void set_struct_arg_impl(std::vector<int> arg_indices, T v);

  template <typename T>
  void set_struct_arg(std::vector<int> arg_indices, T v);

  void set_ndarray_ptrs(int arg_id, uint64 data_ptr, uint64 grad_ptr);

  template <typename T>
  T get_arg(int i);

  template <typename T>
  T get_struct_arg(std::vector<int> arg_indices);

  template <typename T>
  T get_ret(int i);

  void set_arg_external_array_with_shape(int arg_id,
                                         uintptr_t ptr,
                                         uint64 size,
                                         const std::vector<int64> &shape,
                                         uintptr_t grad_ptr = 0);

  void set_arg_ndarray_impl(int arg_id,
                            intptr_t devalloc_ptr,
                            const std::vector<int> &shape,
                            intptr_t devalloc_ptr_grad = 0);
  void set_arg_ndarray(int arg_id, const Ndarray &arr);
  void set_arg_ndarray_with_grad(int arg_id,
                                 const Ndarray &arr,
                                 const Ndarray &arr_grad);

  void set_arg_texture_impl(int arg_id, intptr_t alloc_ptr);
  void set_arg_texture(int arg_id, const Texture &tex);
  void set_arg_rw_texture_impl(int arg_id,
                               intptr_t alloc_ptr,
                               const std::array<int, 3> &shape);
  void set_arg_rw_texture(int arg_id, const Texture &tex);

  void set_arg_matrix(int arg_id, const Matrix &matrix);

  TypedConstant fetch_ret(const std::vector<int> &index);
  float64 get_struct_ret_float(const std::vector<int> &index);
  int64 get_struct_ret_int(const std::vector<int> &index);
  uint64 get_struct_ret_uint(const std::vector<int> &index);

  RuntimeContext &get_context();

 private:
  TypedConstant fetch_ret_impl(int offset, const Type *dt);
  CallableBase *kernel_;
  std::unique_ptr<RuntimeContext> owned_ctx_;
  // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
  // that the caller passes a RuntimeContext pointer externally. In that case,
  // |owned_ctx_| will be nullptr.
  // Invariant: |ctx_| will never be nullptr.
  RuntimeContext *ctx_;
  std::unique_ptr<char[]> arg_buffer_;
  std::unique_ptr<char[]> result_buffer_;
  const StructType *ret_type_;

 public:
  size_t arg_buffer_size{0};
  const StructType *args_type{nullptr};
  size_t result_buffer_size{0};

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  uint64 array_runtime_sizes[taichi_max_num_args_total]{0};
  // `device_allocation_type` is set iff i-th arg is a `DeviceAllocation*`,
  // otherwise it is set to DevAllocType::kNone
  DevAllocType device_allocation_type[taichi_max_num_args_total]{
      DevAllocType::kNone};

  std::
      unordered_map<std::vector<int>, void *, hashing::Hasher<std::vector<int>>>
          array_ptrs;
};

}  // namespace taichi::lang
