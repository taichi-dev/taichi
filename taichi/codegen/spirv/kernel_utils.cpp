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

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : args_bytes_(0),
      rets_bytes_(0),
      extra_args_bytes_(RuntimeContext::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes aa;
    aa.dt = ka.dt;
    const size_t dt_bytes = data_type_size(aa.dt);
    aa.is_array = ka.is_array;
    aa.stride = dt_bytes;
    aa.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    size_t dt_bytes{0};
    if (auto tensor_type = kr.dt->cast<TensorType>()) {
      ra.dt = tensor_type->get_element_type();
      dt_bytes = data_type_size(ra.dt);
      ra.is_array = true;
      ra.stride = tensor_type->get_num_elements() * dt_bytes;
    } else {
      ra.dt = kr.dt;
      dt_bytes = data_type_size(ra.dt);
      ra.is_array = false;
      ra.stride = dt_bytes;
    }
    ra.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(ra);
  }

  auto arange_args = [](auto *vec, size_t offset, bool is_ret) -> size_t {
    size_t bytes = offset;
    for (int i = 0; i < vec->size(); ++i) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes = (attribs.is_array && !is_ret)
                                  ? sizeof(uint64_t)
                                  : data_type_size(attribs.dt);
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += dt_bytes;
      TI_TRACE(
          "  at={} {} offset_in_mem={} stride={}",
          (*vec)[i].is_array ? (is_ret ? "array" : "vector ptr") : "scalar", i,
          attribs.offset_in_mem, attribs.stride);
    }
    return bytes - offset;
  };

  TI_TRACE("args:");
  args_bytes_ = arange_args(&arg_attribs_vec_, 0, false);
  // Align to extra args
  args_bytes_ = (args_bytes_ + 4 - 1) / 4 * 4;

  TI_TRACE("rets:");
  rets_bytes_ = arange_args(&ret_attribs_vec_, 0, true);

  TI_TRACE("sizes: args={} rets={}", args_bytes(), rets_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
