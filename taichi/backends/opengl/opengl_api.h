#pragma once

#include "taichi/common/core.h"
#include "taichi/ir/transforms.h"

#include <string>
#include <vector>
#include <optional>

#include "taichi/backends/device.h"
#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/backends/opengl/opengl_kernel_launcher.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

class Kernel;
class OffloadedStmt;

namespace opengl {

bool initialize_opengl(bool error_tolerance = false);
bool is_opengl_api_available();

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

struct CompiledKernel {
  std::string kernel_name;
  std::string kernel_src;
  int workgroup_size;
  int num_groups;

  TI_IO_DEF(kernel_name, kernel_src, workgroup_size, num_groups);
};

struct CompiledProgram {
  void init_args(Kernel *kernel);
  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_workgrous,
           int workgroup_size,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access =
               nullptr);
  void set_used(const UsedFeature &used);
  int lookup_or_add_string(const std::string &str);

  bool check_ext_arr_read(int i) const;
  bool check_ext_arr_write(int i) const;

  std::vector<CompiledKernel> kernels;

  int arg_count{0};
  int ret_count{0};
  size_t args_buf_size{0};
  size_t total_ext_arr_size{0};
  size_t ret_buf_size{0};

  std::unordered_map<int, size_t> ext_arr_map;
  std::unordered_map<int, irpass::ExternalPtrAccess> ext_arr_access;
  std::vector<std::string> str_table;
  UsedFeature used;

  TI_IO_DEF(kernels,
            arg_count,
            ret_count,
            args_buf_size,
            total_ext_arr_size,
            ret_buf_size,
            ext_arr_map,
            ext_arr_access,
            str_table);
};

class DeviceCompiledProgram {
 public:
  DeviceCompiledProgram(CompiledProgram &&program, Device *device);
  void launch(Context &ctx, OpenGlRuntime *runtime) const;

 private:
  Device *device_;
  CompiledProgram program_;

  std::vector<std::unique_ptr<Pipeline>> compiled_pipeline_;

  DeviceAllocation args_buf_{kDeviceNullAllocation};
  DeviceAllocation ext_arr_buf_{kDeviceNullAllocation};
  DeviceAllocation ret_buf_{kDeviceNullAllocation};
};

}  // namespace opengl

TLANG_NAMESPACE_END
