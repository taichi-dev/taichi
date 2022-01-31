#include "taichi/codegen/spirv/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace spirv {

// static
std::string TaskAttributes::buffers_name(BufferInfo b) {
  if (b.type == BufferType::Context) {
    return "Context";
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

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : args_bytes_(0),
      rets_bytes_(0),
      extra_args_bytes_(RuntimeContext::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes aa;
    aa.dt = ka.dt;
    const size_t dt_bytes = data_type_size(aa.dt);
    /*
    if (dt_bytes > 4) {
      TI_ERROR("SPIRV kernel only supports less than 32-bit arguments, got {}",
               data_type_name(aa.dt));
    }
    */
    aa.is_array = ka.is_array;
    aa.stride = dt_bytes;
    aa.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    ra.dt = kr.dt;
    const size_t dt_bytes = data_type_size(ra.dt);
    if (dt_bytes > 4) {
      // Metal doesn't support 64bit data buffers.
      TI_ERROR(
          "SPIRV kernel only supports less than 32-bit return value, got {}",
          data_type_name(ra.dt));
    }
    ra.is_array = false;  // TODO(#909): this is a temporary limitation
    ra.stride = dt_bytes;
    ra.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(ra);
  }

  auto arrange_scalar_before_array = [](auto *vec, size_t offset) -> size_t {
    std::vector<int> scalar_indices;
    std::vector<int> array_indices;
    size_t bytes = offset;
    for (int i = 0; i < vec->size(); ++i) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes = data_type_size(attribs.dt);
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
      TI_TRACE("  at={} {} offset_in_mem={} stride={}", (*vec)[i].is_array ? "vector ptr" : "scalar", i,
               attribs.offset_in_mem, attribs.stride);
    }
    return bytes - offset;
  };

  TI_TRACE("args:");
  args_bytes_ = arrange_scalar_before_array(&arg_attribs_vec_, 0);
  TI_TRACE("rets:");
  rets_bytes_ = arrange_scalar_before_array(&ret_attribs_vec_, args_bytes_);

  TI_TRACE("sizes: args={} rets={} ctx={} total={}", args_bytes(), rets_bytes(),
           ctx_bytes(), total_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
