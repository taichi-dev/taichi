#pragma once

#include <string>
#include <vector>

#include "taichi/ir/statements.h"
#include "taichi/backends/metal/data_types.h"

// Data structures defined in this file may overlap with some of the Taichi data
// structures. However, they serve as a boundary between Taichi and Metal and
// help decouple the system.
//
// Please keep in mind that each Taichi kernel defined by @ti.kernel can be
// compiled to more than one Metal compute kernels. Concretely, each offloaded
// task in the Taichi kernel maps to a Metal kernel.

TLANG_NAMESPACE_BEGIN

class Kernel;
class SNode;

namespace metal {

// This struct holds the necessary information to launch a Metal kernel.
struct KernelAttributes {
  enum class Buffers {
    Root,
    GlobalTmps,
    Args,
    Runtime,
  };
  std::string name;
  int num_threads;
  OffloadedStmt::TaskType task_type;

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
  };

  struct RuntimeListOpAttributes {
    const SNode *snode = nullptr;
  };
  std::vector<Buffers> buffers;
  // Only valid when |task_type| is range_for.
  // TODO(k-ye): Use std::optional to wrap |task_type| dependent attributes.
  RangeForAttributes range_for_attribs;
  // clear_list + listgen
  RuntimeListOpAttributes runtime_list_op_attribs;

  static std::string buffers_name(Buffers b);
  std::string debug_string() const;
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
    // For array arg, this is #elements * stride(dt). Unit: byte
    size_t stride = 0;
    // Offset in the argument buffer
    size_t offset_in_mem = 0;
    // Index of the input arg or the return value in the host `Context`
    int index = -1;
    MetalDataType dt;
    bool is_array = false;
  };

 public:
  // This is mostly the same as Kernel::Arg, with Metal specific attributes.
  struct ArgAttributes : public AttribsBase {};

  // This is mostly the same as Kernel::Ret, with Metal specific attributes.
  struct RetAttributes : public AttribsBase {};

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

}  // namespace metal

TLANG_NAMESPACE_END
