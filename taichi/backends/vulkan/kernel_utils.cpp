#include "taichi/backends/vulkan/kernel_utils.h"

#include <unordered_map>

#include "taichi/backends/vulkan/data_type_utils.h"
#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace vulkan {

// static
std::string TaskAttributes::buffers_name(Buffers b) {
#define REGISTER_NAME(x) \
  { Buffers::x, #x }
  const static std::unordered_map<Buffers, std::string> m = {
      REGISTER_NAME(Root),
      REGISTER_NAME(GlobalTmps),
      REGISTER_NAME(Context),
  };
#undef REGISTER_NAME
  return m.find(b)->second;
}

std::string TaskAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<TaskAttributes name={} advisory_total_num_threads={} "
      "task_type={} buffers=[ ",
      name, advisory_total_num_threads, offloaded_task_type_name(task_type));
  for (auto b : buffer_binds) {
    result += buffers_name(b.type) + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  result += ">";
  return result;
}

std::string TaskAttributes::BufferBind::debug_string() const {
  return fmt::format("<type={} binding={}>", TaskAttributes::buffers_name(type),
                     binding);
}

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : args_bytes_(0),
      rets_bytes_(0),
      extra_args_bytes_(Context::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes aa;
    aa.dt = ka.dt;
    const size_t dt_bytes = vk_data_type_size(aa.dt);
    if (dt_bytes != 4) {
      TI_ERROR("Vulakn kernel only supports 32-bit data, got {}",
               data_type_name(aa.dt));
    }
    aa.is_array = ka.is_external_array;
    // For array, |ka.size| is #elements * elements_size
    aa.stride = aa.is_array ? ka.size : dt_bytes;
    aa.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    ra.dt = kr.dt;
    const size_t dt_bytes = vk_data_type_size(ra.dt);
    if (dt_bytes != 4) {
      TI_ERROR("Vulakn kernel only supports 32-bit data, got {}",
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
    for (int i = 0; i < vec->size(); ++i) {
      if ((*vec)[i].is_array) {
        array_indices.push_back(i);
      } else {
        scalar_indices.push_back(i);
      }
    }
    size_t bytes = offset;
    // Put scalar args in the memory first
    for (int i : scalar_indices) {
      auto &attribs = (*vec)[i];
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
      TI_TRACE("  at={} scalar offset_in_mem={} stride={}", i,
               attribs.offset_in_mem, attribs.stride);
    }
    // Then the array args
    for (int i : array_indices) {
      auto &attribs = (*vec)[i];
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
      TI_TRACE("  at={} array offset_in_mem={} stride={}", i,
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
  TI_ASSERT(has_args() == (args_bytes_ > 0));
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
