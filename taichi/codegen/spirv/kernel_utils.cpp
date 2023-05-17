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
    if (ka.get_element_type()->is<PrimitiveType>()) {
      aa.dtype = ka.get_element_type()->as<PrimitiveType>()->type;
    } else if (ka.get_element_type()->is<StructType>() && ka.is_array) {
      auto element_type =
          ka.get_element_type()
              ->as<StructType>()
              ->get_element_type({TypeFactory::DATA_PTR_POS_IN_NDARRAY})
              ->as<PointerType>()
              ->get_pointee_type();

      if (element_type->is<TensorType>()) {
        auto tensor_type = element_type->as<TensorType>();
        aa.dtype = tensor_type->get_element_type()->as<PrimitiveType>()->type;
        aa.element_shape = tensor_type->get_shape();
      } else {
        aa.dtype = element_type->as<PrimitiveType>()->type;
      }
    } else {
      // TODO: handle ti.Vector & ti.Matrix
    }
    aa.index = arg_attribs_vec_.size();
    aa.field_dim = ka.total_dim - aa.element_shape.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    size_t dt_bytes{0};
    if (auto tensor_type = kr.dt->cast<TensorType>()) {
      auto tensor_dtype = tensor_type->get_element_type();
      TI_ASSERT(tensor_dtype->is<PrimitiveType>());
      ra.dtype = tensor_dtype->cast<PrimitiveType>()->type;
      dt_bytes = data_type_size(tensor_dtype);
      ra.is_array = true;
      ra.stride = tensor_type->get_num_elements() * dt_bytes;
    } else {
      TI_ASSERT(kr.dt->is<PrimitiveType>());
      ra.dtype = kr.dt->cast<PrimitiveType>()->type;
      dt_bytes = data_type_size(kr.dt);
      ra.is_array = false;
      ra.stride = dt_bytes;
    }
    ra.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(ra);
  }

  auto arange_args = [](auto *vec, size_t offset, bool is_ret,
                        bool has_buffer_ptr) -> size_t {
    size_t bytes = offset;
    for (int i = 0; i < vec->size(); ++i) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes =
          (attribs.is_array && !is_ret)
              ? (has_buffer_ptr ? sizeof(uint64_t) : sizeof(uint32_t))
              : data_type_size(PrimitiveType::get(attribs.dtype));
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += is_ret ? attribs.stride : dt_bytes;
      TI_TRACE(
          "  at={} {} offset_in_mem={} stride={}",
          (*vec)[i].is_array ? (is_ret ? "array" : "vector ptr") : "scalar", i,
          attribs.offset_in_mem, attribs.stride);
    }
    return bytes - offset;
  };

  args_type_ = kernel.args_type;
  rets_type_ = kernel.ret_type;

  args_bytes_ = kernel.args_size;

  TI_TRACE("rets:");
  rets_bytes_ = arange_args(&ret_attribs_vec_, 0, true, false);

  TI_ASSERT(ret_attribs_vec_.size() == kernel.ret_type->elements().size());
  for (int i = 0; i < ret_attribs_vec_.size(); ++i) {
    TI_ASSERT(ret_attribs_vec_[i].offset_in_mem ==
              kernel.ret_type->get_element_offset({i}));
  }

  TI_ASSERT(rets_bytes_ == kernel.ret_size);

  TI_TRACE("sizes: args={} rets={}", args_bytes(), rets_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace taichi::lang
