#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/backends/metal/data_types.h"

// Data structures defined in this file may overlap with some of the Taichi data
// structures. However, they serve as a boundary between Taichi and Metal and
// help decouple the system.
//
// Please keep in mind that each Taichi kernel defined by @ti.kernel can be
// compiled to more than one Metal compute kernels. Concretely, each offloaded
// task in the Taichi kernel maps to a Metal kernel.

namespace taichi {
namespace lang {

class Kernel;
class SNode;

namespace metal {

// TODO(k-ye): Share this between OpenGL and Metal?
class PrintStringTable {
 public:
  int put(const std::string &str);
  const std::string &get(int i);

 private:
  std::vector<std::string> strs_;
};

struct BufferDescriptor {
  enum class Type {
    Root,
    GlobalTmps,
    Context,
    Runtime,
    Print,
    Ndarray,
  };

  BufferDescriptor() = default;

  static BufferDescriptor root(int root_id) {
    return BufferDescriptor{Type::Root, root_id};
  }

  static BufferDescriptor global_tmps() {
    return BufferDescriptor{Type::GlobalTmps};
  }

  static BufferDescriptor context() {
    return BufferDescriptor{Type::Context};
  }

  static BufferDescriptor runtime() {
    return BufferDescriptor{Type::Runtime};
  }

  static BufferDescriptor print() {
    return BufferDescriptor{Type::Print};
  }

  static BufferDescriptor ndarray(int arr_arg_id) {
    return BufferDescriptor{Type::Ndarray, arr_arg_id};
  }

  Type type() const {
    return type_;
  }

  int root_id() const {
    TI_ASSERT(type_ == Type::Root);
    return id_;
  }

  int ndarray_arg_id() const {
    TI_ASSERT(type_ == Type::Ndarray);
    return id_;
  }

  std::string debug_string() const;

  bool operator==(const BufferDescriptor &other) const;

  bool operator!=(const BufferDescriptor &other) const {
    return !(*this == other);
  }

  struct Hasher {
    std::size_t operator()(const BufferDescriptor &desc) const {
      return std::hash<BufferDescriptor::Type>{}(desc.type()) ^ desc.id_;
    }
  };

 private:
  explicit BufferDescriptor(Type t) : type_(t) {
  }

  explicit BufferDescriptor(Type t, int root_id) : type_(t), id_(root_id) {
  }
  Type type_{Type::Root};
  int id_{-1};  // only used if type in {Root, Ndarray}

 public:
  TI_IO_DEF(type_, id_);
};

// This struct holds the necessary information to launch a Metal kernel.
struct KernelAttributes {
  std::string name;
  // Total number of threads to launch (i.e. threads per grid). Note that this
  // is only advisory, because eventually this number is also determined by the
  // runtime config. This works because grid strided loop is supported.
  int advisory_total_num_threads;
  // Block size in CUDA's terminology. On Metal, it is called a threadgroup.
  int advisory_num_threads_per_group;

  OffloadedTaskType task_type;

  struct RangeForAttributes {
    // |begin| has differen meanings depending on |const_begin|:
    // * true : It is the left boundary of the loop known at compile time.
    // * false: It is the offset of the begin in the global tmps buffer.
    //
    // Same applies to |end|.
    size_t begin;
    size_t end;
    bool const_begin{true};
    bool const_end{true};

    inline bool const_range() const {
      return (const_begin && const_end);
    }

    TI_IO_DEF(begin, end, const_begin, const_end);
  };

  struct RuntimeListOpAttributes {
    const SNode *snode = nullptr;
  };
  struct GcOpAttributes {
    const SNode *snode = nullptr;
  };
  std::vector<BufferDescriptor> buffers;
  std::unordered_map<int, int> arr_args_to_binding_indices;
  // Only valid when |task_type| is `range_for`.
  std::optional<RangeForAttributes> range_for_attribs;
  // Only valid when |task_type| is `listgen`.
  std::optional<RuntimeListOpAttributes> runtime_list_op_attribs;
  // Only valid when |task_type| is `gc`.
  std::optional<GcOpAttributes> gc_op_attribs;

  std::string debug_string() const;

  TI_IO_DEF(name,
            advisory_total_num_threads,
            task_type,
            buffers,
            range_for_attribs);
};

// Groups all the Metal kernels generated from a single ti.kernel
struct TaichiKernelAttributes {
  struct UsedFeatures {
    // Whether print() is called inside this kernel.
    bool print = false;
    // Whether assert is called inside this kernel.
    bool assertion = false;
    // Whether this kernel accesses (read or write) sparse SNodes.
    bool sparse = false;
    // Whether [[thread_index_in_simdgroup]] is used. This is only supported
    // since MSL 2.1
    bool simdgroup = false;
  };
  std::string name;
  // Is this kernel for evaluating the constant fold result?
  bool is_jit_evaluator = false;
  // Attributes of all the Metal kernels produced from this Taichi kernel.
  std::vector<KernelAttributes> mtl_kernels_attribs;
  UsedFeatures used_features;

  TI_IO_DEF(name, mtl_kernels_attribs);
};

// This class contains the attributes descriptors for both the input args and
// the return values of a Taichi kernel.
//
// Note that all Metal kernels belonging to the same Taichi kernel will share
// the same kernel args (i.e. they use the same Metal buffer for input args and
// return values). This is because kernel arguments is a Taichi-level concept.
class KernelContextAttributes {
 private:
  // Attributes that are shared by the input arg and the return value.
  struct AttribsBase {
    // This is tricky:
    // * For Args
    //    * scalar: stride(dt)
    //    * array: 0
    // * For Return, this can actually be a matrix, where `is_array` is true...
    // Unit: byte.
    size_t stride = 0;
    // Offset in the argument buffer
    size_t offset_in_mem = 0;
    // Index of the input arg or the return value in the host `Context`
    int index = -1;
    MetalDataType dt;
    bool is_array = false;

    TI_IO_DEF(stride, offset_in_mem, index, dt, is_array);
  };

 public:
  // This is mostly the same as Kernel::Arg, with Metal specific attributes.
  struct ArgAttributes : public AttribsBase {};

  // This is mostly the same as Kernel::Ret, with Metal specific attributes.
  struct RetAttributes : public AttribsBase {};

  KernelContextAttributes() = default;
  explicit KernelContextAttributes(const Kernel &kernel);

  inline bool has_args() const {
    return !arg_attribs_vec_.empty();
  }

  inline const std::vector<ArgAttributes> &args() const {
    return arg_attribs_vec_;
  }

  inline bool has_rets() const {
    return !ret_attribs_vec_.empty();
  }

  inline const std::vector<RetAttributes> &rets() const {
    return ret_attribs_vec_;
  }

  // Returns true if the kernel has neither input args nor return values.
  inline bool empty() const {
    return !(has_args() || has_rets());
  }

  // Total size in bytes of the input args and return values
  inline size_t ctx_bytes() const {
    return ctx_bytes_;
  }
  inline size_t extra_args_bytes() const {
    return extra_args_bytes_;
  }
  // Total bytes needed for allocating the Metal buffer
  inline size_t total_bytes() const {
    return ctx_bytes_ + extra_args_bytes_;
  }

  TI_IO_DEF(arg_attribs_vec_, ret_attribs_vec_, ctx_bytes_, extra_args_bytes_);

 private:
  // Memory layout
  //
  // /---- input args ----\/---- ret vals -----\/-- extra args --\
  // +----------+---------+----------+---------+-----------------+
  // |  scalar  |  array  |  scalar  |  array  |      scalar     |
  // +----------+---------+----------+---------+-----------------+
  //
  std::vector<ArgAttributes> arg_attribs_vec_;
  std::vector<RetAttributes> ret_attribs_vec_;
  // Total size in bytes of the input args and return values
  size_t ctx_bytes_;
  size_t extra_args_bytes_;
};

struct CompiledKernelData {
  std::string kernel_name;
  std::string source_code;
  KernelContextAttributes ctx_attribs;
  TaichiKernelAttributes kernel_attribs;

  TI_IO_DEF(kernel_name, ctx_attribs, kernel_attribs);
};

struct CompiledKernelTmplData {
  std::string kernel_bundle_name;
  std::unordered_map<std::string, CompiledKernelData> kernel_tmpl_map;

  TI_IO_DEF(kernel_bundle_name, kernel_tmpl_map);
};

struct CompiledFieldData {
  std::string field_name;
  MetalDataType dtype;
  std::string dtype_name;
  std::vector<int> shape;
  int mem_offset_in_parent{0};
  bool is_scalar{false};
  int row_num{0};
  int column_num{0};

  TI_IO_DEF(field_name,
            dtype,
            dtype_name,
            shape,
            mem_offset_in_parent,
            is_scalar,
            row_num,
            column_num);
};

struct BufferMetaData {
  int64_t root_buffer_size{0};
  int64_t runtime_buffer_size{0};
  int64_t randseedoffset_in_runtime_buffer{0};

  TI_IO_DEF(root_buffer_size,
            runtime_buffer_size,
            randseedoffset_in_runtime_buffer);
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
