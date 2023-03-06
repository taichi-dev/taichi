#pragma once
#include <taichi/program/callable.h>
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"

namespace taichi::lang {

class LaunchContextBuilder {
 public:
  LaunchContextBuilder(CallableBase *kernel, RuntimeContext *ctx);
  explicit LaunchContextBuilder(CallableBase *kernel);

  LaunchContextBuilder(LaunchContextBuilder &&) = default;
  LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
  LaunchContextBuilder(const LaunchContextBuilder &) = delete;
  LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

  void set_arg_float(int arg_id, float64 d);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(int arg_id, int64 d);
  void set_arg_uint(int arg_id, uint64 d);

  void set_extra_arg_int(int i, int j, int32 d);

  void set_arg_external_array_with_shape(int arg_id,
                                         uintptr_t ptr,
                                         uint64 size,
                                         const std::vector<int64> &shape);

  void set_arg_ndarray(int arg_id, const Ndarray &arr);
  void set_arg_ndarray_with_grad(int arg_id,
                                 const Ndarray &arr,
                                 const Ndarray &arr_grad);

  void set_arg_texture(int arg_id, const Texture &tex);
  void set_arg_rw_texture(int arg_id, const Texture &tex);

  // Sets the |arg_id|-th arg in the context to the bits stored in |d|.
  // This ignores the underlying kernel's |arg_id|-th arg type.
  void set_arg_raw(int arg_id, uint64 d);
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
};

}  // namespace taichi::lang
