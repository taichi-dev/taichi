#include "taichi/backends/metal/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {

namespace metal {

int PrintStringTable::put(const std::string &str) {
  int i = 0;
  for (; i < strs_.size(); ++i) {
    if (str == strs_[i]) {
      return i;
    }
  }
  strs_.push_back(str);
  return i;
}

const std::string &PrintStringTable::get(int i) {
  return strs_[i];
}

bool BufferDescriptor::operator==(const BufferDescriptor &other) const {
  if (type_ != other.type_) {
    return false;
  }
  if (type_ == Type::Root) {
    return root_id_ == other.root_id_;
  }
  TI_ASSERT(root_id_ == -1);
  return true;
}

std::string BufferDescriptor::debug_string() const {
#define REGISTER_NAME(x) \
  { Type::x, #x }
  const static std::unordered_map<Type, std::string> m = {
      REGISTER_NAME(GlobalTmps),
      REGISTER_NAME(Context),
      REGISTER_NAME(Runtime),
      REGISTER_NAME(Print),
  };
#undef REGISTER_NAME
  if (type_ == Type::Root) {
    return fmt::format("Root_{}", root_id());
  }
  return m.find(type_)->second;
}

std::string KernelAttributes::debug_string() const {
  std::string result;
  result += fmt::format(
      "<KernelAttributes name={} num_threads={} num_threads_per_group={} "
      "task_type={} buffers=[ ",
      name, advisory_total_num_threads, advisory_num_threads_per_group,
      offloaded_task_type_name(task_type));
  for (auto b : buffers) {
    result += b.debug_string() + " ";
  }
  result += "]";  // closes |buffers|
  // TODO(k-ye): show range_for
  if (task_type == OffloadedTaskType::listgen) {
    result += fmt::format(" snode={}", runtime_list_op_attribs->snode->id);
  } else if (task_type == OffloadedTaskType::gc) {
    result += fmt::format(" snode={}", gc_op_attribs->snode->id);
  }
  result += ">";
  return result;
}

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : ctx_bytes_(0), extra_args_bytes_(RuntimeContext::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes ma;
    ma.dt = to_metal_type(ka.dt);
    const size_t dt_bytes = metal_data_type_bytes(ma.dt);
    if (dt_bytes > 4) {
      // Metal doesn't support 64bit data buffers.
      TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
               metal_data_type_name(ma.dt));
    }
    ma.is_array = ka.is_external_array;
    ma.stride = ma.is_array ? ka.size : dt_bytes;
    ma.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(ma);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes mr;
    mr.dt = to_metal_type(kr.dt);
    const size_t dt_bytes = metal_data_type_bytes(mr.dt);
    if (dt_bytes > 4) {
      // Metal doesn't support 64bit data buffers.
      TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
               metal_data_type_name(mr.dt));
    }
    mr.is_array = false;  // TODO(#909): this is a temporary limitation
    mr.stride = dt_bytes;
    mr.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(mr);
  }

  auto arrange_scalar_before_array = [&bytes = this->ctx_bytes_](auto *vec) {
    std::vector<int> scalar_indices;
    std::vector<int> array_indices;
    for (int i = 0; i < vec->size(); ++i) {
      if ((*vec)[i].is_array) {
        array_indices.push_back(i);
      } else {
        scalar_indices.push_back(i);
      }
    }
    // Put scalar args in the memory first
    for (int i : scalar_indices) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes = metal_data_type_bytes(attribs.dt);
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
    }
    // Then the array args
    for (int i : array_indices) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes = metal_data_type_bytes(attribs.dt);
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
    }
  };

  arrange_scalar_before_array(&arg_attribs_vec_);
  arrange_scalar_before_array(&ret_attribs_vec_);
}

}  // namespace metal

}  // namespace lang
}  // namespace taichi
