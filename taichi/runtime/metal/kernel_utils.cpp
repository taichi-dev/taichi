#include "taichi/runtime/metal/kernel_utils.h"

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

  return id_ == other.id_;
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
  if (type_ == Type::Ndarray) {
    return fmt::format("Ndarray_{}", ndarray_arg_id());
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
    ma.dt = to_metal_type(ka.get_element_type());
    const size_t dt_bytes = metal_data_type_bytes(ma.dt);
    if (dt_bytes > 4) {
      // Metal doesn't support 64bit data buffers.
      TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
               metal_data_type_name(ma.dt));
    }
    ma.is_array = ka.is_array;
    ma.stride = ma.is_array ? 0 : dt_bytes;
    ma.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(ma);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes mr;
    if (auto tensor_type = kr.dt->cast<TensorType>()) {
      mr.dt = to_metal_type(tensor_type->get_element_type());
      const size_t dt_bytes = metal_data_type_bytes(mr.dt);
      mr.is_array = true;
      if (dt_bytes > 4) {
        // Metal doesn't support 64bit data buffers.
        TI_ERROR(
            "Metal kernel only supports <= 32-bit data, got {} which is "
            "Tensor's element type",
            metal_data_type_name(mr.dt));
      }
      mr.stride =
          tensor_type->get_num_elements() * metal_data_type_bytes(mr.dt);
    } else {
      mr.dt = to_metal_type(kr.dt);
      const size_t dt_bytes = metal_data_type_bytes(mr.dt);
      mr.is_array = false;
      if (dt_bytes > 4) {
        // Metal doesn't support 64bit data buffers.
        TI_ERROR("Metal kernel only supports <= 32-bit data, got {}",
                 metal_data_type_name(mr.dt));
      }
      mr.stride = metal_data_type_bytes(mr.dt);
    }
    mr.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(mr);
  }

  auto arrange_scalar_before_array = [&bytes = this->ctx_bytes_](
                                         auto *vec, bool allow_arr_mem_offset) {
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
      if (allow_arr_mem_offset) {
        const size_t dt_bytes = metal_data_type_bytes(attribs.dt);
        bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
        attribs.offset_in_mem = bytes;
        bytes += attribs.stride;
      } else {
        // Array args are no longer embedded, they have dedicated MTLBuffers.
        attribs.offset_in_mem = -1;
      }
    }
  };

  arrange_scalar_before_array(&arg_attribs_vec_,
                              /*allow_arr_mem_offset=*/false);
  arrange_scalar_before_array(&ret_attribs_vec_, /*allow_arr_mem_offset=*/true);
}

}  // namespace metal

}  // namespace lang
}  // namespace taichi
