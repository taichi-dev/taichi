#include "taichi/codegen/spirv/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace spirv {

// static
std::string TaskAttributes::buffers_name(BufferInfo b) {
  if (b.type == BufferType::Args) {
    return "Args";
  }
  if (b.type == BufferType::Rets) {
    return "Rets";
  }
  if (b.type == BufferType::GlobalTmps) {
    return "GlobalTmps";
  }
  if (b.type == BufferType::Root) {
    return std::string("Root: ") + std::to_string(b.root_id);
  }
  TI_ERROR("unrecognized buffer type");
}

std::string TaskAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<TaskAttributes name={} advisory_total_num_threads={} "
      "task_type={} buffers=[ ",
      name, advisory_total_num_threads, offloaded_task_type_name(task_type));
  for (auto b : buffer_binds) {
    result += buffers_name(b.buffer) + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  result += ">";
  return result;
}

std::string TaskAttributes::BufferBind::debug_string() const {
  return fmt::format("<type={} binding={}>",
                     TaskAttributes::buffers_name(buffer), binding);
}

KernelContextAttributes::KernelContextAttributes(
    const Kernel &kernel,
    const DeviceCapabilityConfig *caps)
    : args_bytes_(0), rets_bytes_(0) {
  arr_access.resize(kernel.parameter_list.size(), irpass::ExternalPtrAccess(0));
  arg_attribs_vec_.reserve(kernel.parameter_list.size());
  // TODO: We should be able to limit Kernel args and rets to be primitive types
  // as well but let's leave that as a followup up PR.
  for (const auto &ka : kernel.parameter_list) {
    ArgAttributes aa;
    aa.name = ka.name;
    aa.is_array = ka.is_array;
    arg_attribs_vec_.push_back(aa);
  }
  // TODO:
  //  ret_attribs_vec_ and this for loop is redundant now. Remove it in a follow
  //  up PR. We keep this loop and use i32 as a placeholder to ensure that
  //  GfxRuntime::device_to_host::require_sync works properly.
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    ra.dtype = PrimitiveTypeID::i32;
    ret_attribs_vec_.push_back(ra);
  }

  args_type_ = kernel.args_type;
  rets_type_ = kernel.ret_type;

  args_bytes_ = kernel.args_size;
  rets_bytes_ = kernel.ret_size;

  TI_TRACE("sizes: args={} rets={}", args_bytes(), rets_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace taichi::lang
