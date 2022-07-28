#pragma once

#include <optional>
#include <string>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/type.h"
#include "taichi/ir/transforms.h"
#include "taichi/rhi/device.h"

namespace taichi {
namespace lang {

class Kernel;
class SNode;

namespace spirv {

/**
 * Per offloaded task attributes.
 */
struct TaskAttributes {
  enum class BufferType { Root, GlobalTmps, Args, Rets, ListGen, ExtArr };

  struct BufferInfo {
    BufferType type;
    int root_id{-1};  // only used if type==Root or type==ExtArr

    BufferInfo() = default;

    BufferInfo(BufferType buffer_type) : type(buffer_type) {
    }

    BufferInfo(BufferType buffer_type, int root_buffer_id)
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

      return hash<BufferType>()(buf.type) ^ buf.root_id;
    }
  };

  struct BufferBind {
    BufferInfo buffer;
    int binding{0};

    std::string debug_string() const;

    TI_IO_DEF(buffer, binding);
  };

  struct TextureBind {
    int arg_id{0};
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
    // For scalar arg, this is max(stride(dt), 4)
    // For array arg, this is #elements * max(stride(dt), 4)
    // Unit: byte
    size_t stride{0};
    // Offset in the context buffer
    size_t offset_in_mem{0};
    // Index of the input arg or the return value in the host `Context`
    int index{-1};
    PrimitiveTypeID dtype{PrimitiveTypeID::unknown};
    bool is_array{false};
    std::vector<int> element_shape;
    std::size_t field_dim{0};

    TI_IO_DEF(stride,
              offset_in_mem,
              index,
              dtype,
              is_array,
              element_shape,
              field_dim);
  };

 public:
  /**
   * This is mostly the same as Kernel::Arg, with device specific attributes.
   */
  struct ArgAttributes : public AttribsBase {};

  /**
   * This is mostly the same as Kernel::Ret, with device specific attributes.
   */
  struct RetAttributes : public AttribsBase {};

  KernelContextAttributes() = default;
  explicit KernelContextAttributes(const Kernel &kernel, Device *device);

  /**
   * Whether this kernel has any argument
   */
  inline bool has_args() const {
    return !arg_attribs_vec_.empty();
  }

  inline const std::vector<ArgAttributes> &args() const {
    return arg_attribs_vec_;
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
   * Number of bytes needed by the extra arguments.
   *
   * Extra argument region is used to store some metadata, like the shape of the
   * external array.
   */
  inline size_t extra_args_bytes() const {
    return extra_args_bytes_;
  }

  /**
   * Offset (in bytes) of the extra arguments in the memory.
   */
  inline size_t extra_args_mem_offset() const {
    return args_bytes();
  }

  std::vector<irpass::ExternalPtrAccess> arr_access;

  TI_IO_DEF(arg_attribs_vec_,
            ret_attribs_vec_,
            args_bytes_,
            rets_bytes_,
            extra_args_bytes_,
            arr_access);

 private:
  std::vector<ArgAttributes> arg_attribs_vec_;
  std::vector<RetAttributes> ret_attribs_vec_;

  size_t args_bytes_{0};
  size_t rets_bytes_{0};
  size_t extra_args_bytes_{0};
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
}  // namespace lang
}  // namespace taichi
