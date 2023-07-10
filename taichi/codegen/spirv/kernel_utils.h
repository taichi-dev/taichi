#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/type.h"
#include "taichi/ir/transforms.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class Kernel;
class SNode;

namespace spirv {

/**
 * Per offloaded task attributes.
 */
struct TaskAttributes {
  enum class BufferType {
    Root,
    GlobalTmps,
    Args,
    Rets,
    ListGen,
    ExtArr,
    ArgPack
  };

  struct BufferInfo {
    BufferType type;
    std::vector<int> root_id{-1};  // only used if type==Root or type==ExtArr

    BufferInfo() = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    BufferInfo(BufferType buffer_type) : type(buffer_type) {
    }

    BufferInfo(BufferType buffer_type, int root_buffer_id)
        : type(buffer_type), root_id({root_buffer_id}) {
    }

    BufferInfo(BufferType buffer_type, const std::vector<int> &root_buffer_id)
        : type(buffer_type), root_id(root_buffer_id) {
    }

    bool operator==(const BufferInfo &other) const {
      if (type != other.type) {
        return false;
      }
      if (type == BufferType::Root || type == BufferType::ExtArr) {
        return root_id == other.root_id;
      }
      return true;
    }

    TI_IO_DEF(type, root_id);
  };

  struct BufferInfoHasher {
    std::size_t operator()(const BufferInfo &buf) const {
      using std::hash;
      using std::size_t;
      using std::string;

      size_t hash_result = hash<BufferType>()(buf.type);
      for (const int &element : buf.root_id)
        hash_result ^= element;
      return hash_result;
    }
  };

  struct BufferBind {
    BufferInfo buffer;
    int binding{0};

    std::string debug_string() const;

    TI_IO_DEF(buffer, binding);
  };

  struct TextureBind {
    std::vector<int> arg_id;
    int binding{0};
    bool is_storage{false};

    TI_IO_DEF(arg_id, binding, is_storage);
  };

  std::string name;
  std::string source_path;
  // Total number of threads to launch (i.e. threads per grid). Note that this
  // is only advisory, because eventually this number is also determined by the
  // runtime config. This works because grid strided loop is supported.
  int advisory_total_num_threads{0};
  int advisory_num_threads_per_group{0};

  OffloadedTaskType task_type;

  struct RangeForAttributes {
    // |begin| has different meanings depending on |const_begin|:
    // * true : It is the left boundary of the loop known at compile time.
    // * false: It is the offset of the begin in the global tmps buffer.
    //
    // Same applies to |end|.
    size_t begin{0};
    size_t end{0};
    bool const_begin{true};
    bool const_end{true};

    inline bool const_range() const {
      return (const_begin && const_end);
    }

    TI_IO_DEF(begin, end, const_begin, const_end);
  };
  std::vector<BufferBind> buffer_binds;
  std::vector<TextureBind> texture_binds;
  // Only valid when |task_type| is range_for.
  std::optional<RangeForAttributes> range_for_attribs;

  static std::string buffers_name(BufferInfo b);

  std::string debug_string() const;

  TI_IO_DEF(name,
            advisory_total_num_threads,
            advisory_num_threads_per_group,
            task_type,
            buffer_binds,
            texture_binds,
            range_for_attribs);
};

/**
 * This class contains the attributes descriptors for both the input args and
 * the return values of a Taichi kernel.
 *
 * Note that all SPIRV tasks (shaders) belonging to the same Taichi kernel will
 * share the same kernel args (i.e. they use the same device buffer for input
 * args and return values). This is because kernel arguments is a Taichi-level
 * concept.
 *
 * Memory layout
 *
 * /---- input args ----\/---- ret vals -----\/-- extra args --\
 * +----------+---------+----------+---------+-----------------+
 * |  scalar  |  array  |  scalar  |  array  |      scalar     |
 * +----------+---------+----------+---------+-----------------+
 */
class KernelContextAttributes {
 private:
  /**
   * Attributes that are shared by the input arg and the return value.
   */
  struct AttribsBase {
    std::string name;
    // For scalar arg, this is max(stride(dt), 4)
    // For array arg, this is #elements * max(stride(dt), 4)
    // Unit: byte
    size_t stride{0};
    // Offset in the context buffer
    size_t offset_in_mem{0};
    PrimitiveTypeID dtype{PrimitiveTypeID::unknown};
    bool is_array{false};
    std::vector<int> element_shape;
    std::size_t field_dim{0};
    // Only used with textures. Sampled textures always have unknown format;
    // while RW textures always have a valid format.
    BufferFormat format{BufferFormat::unknown};
    ParameterType ptype{ParameterType::kUnknown};

    TI_IO_DEF(name,
              stride,
              offset_in_mem,
              dtype,
              is_array,
              element_shape,
              field_dim,
              format,
              ptype);
  };

 public:
  /**
   * This is mostly the same as Kernel::Arg, with device specific attributes.
   */
  struct ArgAttributes : public AttribsBase {
    // Indices of the arg value in the host `Context`.
    std::vector<int> indices;
    bool is_argpack{false};

    TI_IO_DEF(name,
              stride,
              offset_in_mem,
              indices,
              dtype,
              is_array,
              is_argpack,
              element_shape,
              field_dim,
              format,
              ptype);
  };

  /**
   * This is mostly the same as Kernel::Ret, with device specific attributes.
   */
  struct RetAttributes : public AttribsBase {
    // Index of the return value in the host `Context`.
    int index{-1};

    TI_IO_DEF(name,
              stride,
              offset_in_mem,
              index,
              dtype,
              is_array,
              element_shape,
              field_dim,
              format,
              ptype);
  };

  KernelContextAttributes() = default;
  explicit KernelContextAttributes(const Kernel &kernel,
                                   const DeviceCapabilityConfig *caps);

  /**
   * Whether this kernel has any argument
   */
  inline bool has_args() const {
    return !arg_attribs_vec_.empty();
  }

  inline const std::vector<std::pair<std::vector<int>, ArgAttributes>> &args()
      const {
    return arg_attribs_vec_;
  }

  inline const ArgAttributes &arg_at(const std::vector<int> &indices) const {
    for (const auto &element : arg_attribs_vec_) {
      if (element.first == indices) {
        return element.second;
      }
    }
    TI_ERROR(fmt::format(
        "Unexpected error: ArgAttributes with indices ({}) not found.",
        fmt::join(indices, ", ")));
    return arg_attribs_vec_[0].second;
  }

  /**
   * Whether this kernel has any return value
   */
  inline bool has_rets() const {
    return !ret_attribs_vec_.empty();
  }

  inline const std::vector<RetAttributes> &rets() const {
    return ret_attribs_vec_;
  }

  /**
   * Whether this kernel has either arguments or return values.
   */
  inline bool empty() const {
    return !(has_args() || has_rets());
  }

  /**
   * Number of bytes needed by all the arguments.
   */
  inline size_t args_bytes() const {
    return args_bytes_;
  }

  /**
   * Number of bytes needed by all the return values.
   */
  inline size_t rets_bytes() const {
    return rets_bytes_;
  }

  /**
   * The type of the struct that contains all the arguments.
   */
  inline const lang::StructType *args_type() const {
    return args_type_;
  }

  /**
   * The type of the struct that contains all the return values.
   */
  inline const lang::StructType *rets_type() const {
    return rets_type_;
  }

  /**
   * Get the type of argpack by arg_id.
   */
  inline const lang::Type *argpack_type(const std::vector<int> &arg_id) const {
    for (const auto &element : argpack_types_) {
      if (element.first == arg_id) {
        return element.second;
      }
    }
    return nullptr;
  }

  /**
   * Get all argpacks.
   */
  inline const std::vector<std::pair<std::vector<int>, const Type *>>
      &argpack_types() const {
    return argpack_types_;
  }

  std::vector<std::pair<std::vector<int>, irpass::ExternalPtrAccess>>
      arr_access;

  TI_IO_DEF(arg_attribs_vec_,
            ret_attribs_vec_,
            args_bytes_,
            rets_bytes_,
            arr_access,
            args_type_,
            rets_type_,
            argpack_types_);

 private:
  std::vector<std::pair<std::vector<int>, ArgAttributes>> arg_attribs_vec_;
  std::vector<RetAttributes> ret_attribs_vec_;

  size_t args_bytes_{0};
  size_t rets_bytes_{0};

  const lang::StructType *args_type_{nullptr};
  const lang::StructType *rets_type_{nullptr};

  std::vector<std::pair<std::vector<int>, const Type *>> argpack_types_;
};

/**
 * Groups all the device kernels generated from a single ti.kernel.
 */
struct TaichiKernelAttributes {
  // Taichi kernel name
  std::string name;
  // Is this kernel for evaluating the constant fold result?
  bool is_jit_evaluator{false};
  // Attributes of all the tasks produced from this single Taichi kernel.
  std::vector<TaskAttributes> tasks_attribs;

  KernelContextAttributes ctx_attribs;

  TI_IO_DEF(name, is_jit_evaluator, tasks_attribs, ctx_attribs);
};

}  // namespace spirv
}  // namespace taichi::lang
