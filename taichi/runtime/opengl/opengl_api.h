#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/backends/device.h"
#include "taichi/runtime/opengl/opengl_kernel_launcher.h"
#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/common/core.h"
#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/transforms.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool use_gles = false, bool error_tolerance = false);
bool is_opengl_api_available(bool use_gles = false);
bool is_gles();

#define PER_OPENGL_EXTENSION(x) extern bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

extern int opengl_threads_per_block;

#define TI_OPENGL_REQUIRE(used, x) \
  ([&]() {                         \
    if (opengl_extension_##x) {    \
      used.extension_##x = true;   \
      return true;                 \
    }                              \
    return false;                  \
  })()

struct CompiledOffloadedTask {
  std::string name;
  std::string src;
  OffloadedTaskType type;
  std::string range_hint;
  int workgroup_size;
  int num_groups;

  TI_IO_DEF(name, src, workgroup_size, num_groups);
};

struct ScalarArg {
  std::string dtype_name;
  size_t offset_in_bytes_in_args_buf{0};

  TI_IO_DEF(offset_in_bytes_in_args_buf);
};

struct CompiledArrayArg {
  uint32_t dtype;
  std::string dtype_name;
  std::size_t field_dim{0};
  bool is_scalar{false};
  std::vector<int> element_shape;
  size_t shape_offset_in_bytes_in_args_buf{0};
  size_t runtime_size{0};  // Runtime information

  TI_IO_DEF(field_dim,
            is_scalar,
            element_shape,
            shape_offset_in_bytes_in_args_buf);
};

struct CompiledTaichiKernel {
  void init_args(Kernel *kernel);
  void add(const std::string &name,
           const std::string &source_code,
           OffloadedTaskType type,
           const std::string &range_hint,
           int num_workgrous,
           int workgroup_size,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access =
               nullptr);
  void set_used(const UsedFeature &used);
  int lookup_or_add_string(const std::string &str);

  bool check_ext_arr_read(int i) const;
  bool check_ext_arr_write(int i) const;

  std::vector<CompiledOffloadedTask> tasks;

  int arg_count{0};
  int ret_count{0};
  size_t args_buf_size{0};
  size_t ret_buf_size{0};

  std::unordered_map<int, irpass::ExternalPtrAccess> ext_arr_access;
  std::vector<std::string> str_table;
  UsedFeature used;
  std::unordered_map<int, ScalarArg> scalar_args;
  mutable std::unordered_map<int, CompiledArrayArg> arr_args;

  TI_IO_DEF(tasks,
            arg_count,
            ret_count,
            args_buf_size,
            ret_buf_size,
            scalar_args,
            arr_args,
            used.arr_arg_to_bind_idx);
};

class DeviceCompiledTaichiKernel {
 public:
  DeviceCompiledTaichiKernel(CompiledTaichiKernel &&program, Device *device);
  void launch(RuntimeContext &ctx,
              Kernel *kernel,
              OpenGlRuntime *runtime) const;

 private:
  Device *device_;
  CompiledTaichiKernel program_;

  std::vector<std::unique_ptr<Pipeline>> compiled_pipeline_;

  mutable std::unique_ptr<DeviceAllocationGuard> args_buf_{nullptr};
  DeviceAllocation ret_buf_{kDeviceNullAllocation};
  // Only saves numpy/torch cpu based external array since they don't have
  // DeviceAllocation.
  // Taichi |Ndarray| manages their own DeviceAllocation so it's not saved here.
  mutable DeviceAllocation ext_arr_bufs_[taichi_max_num_args]{
      kDeviceNullAllocation};
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
